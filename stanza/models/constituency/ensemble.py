"""
Prototype of ensembling N models together on the same dataset

The main process is to run the normal transition sequence, but sum the
scores for the N models and use that to choose the highest scoring
transition

Currently the code is kind of awkward because it includes a lot of
duplicated logic in predict() and parse_sentences()

Example of how to run it to build a silver dataset:

python3 stanza/models/constituency/ensemble.py
  saved_models/constituency/en_wsj_inorder_?.pt
   --mode parse_text
   --tokenized_file /nlp/scr/horatio/en_silver/en_split_100
   --predict_file /nlp/scr/horatio/en_silver/en_split_100.inorder.mrg
   --retag_package en_combined_bert
   --lang en

then, ideally, run a second time with a set of topdown models,
then take the trees which match from the files
"""


import argparse
import logging
import os

import torch
import torch.nn as nn

from stanza.models.common import utils
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.constituency import retagging
from stanza.models.constituency import tree_reader
# TODO: move run_dev_set elsewhere or move its usage in this file elsewhere
# same with parse_text & parse_dir
# otherwise there will be circular imports
from stanza.models.constituency.base_trainer import BaseTrainer, ModelType
from stanza.models.constituency.parser_training import run_dev_set
from stanza.models.constituency.state import MultiState
from stanza.models.constituency.text_processing import parse_text, parse_dir
from stanza.models.constituency.trainer import Trainer
from stanza.models.constituency.utils import add_predict_output_args, postprocess_predict_output_args, retag_trees
from stanza.resources.common import DEFAULT_MODEL_DIR
from stanza.server.parser_eval import EvaluateParser, ParseResult, ScoredTree
from stanza.utils.default_paths import get_default_paths

logger = logging.getLogger('stanza.constituency.trainer')

class Ensemble(nn.Module):
    def __init__(self, args, filenames=None, models=None, foundation_cache=None):
        """
        Loads each model in filenames

        If foundation_cache is None, we build one on our own,
        as the expectation is the models will reuse modules
        such as pretrain, charlm, bert
        """
        super().__init__()

        if filenames:
            if foundation_cache is None:
                foundation_cache = FoundationCache()

            if isinstance(filenames, str):
                filenames = [filenames]
            logger.info("Models used for ensemble:\n  %s", "\n  ".join(filenames))
            models = [Trainer.load(filename, args, load_optimizer=False, foundation_cache=foundation_cache).model for filename in filenames]
        elif not models:
            raise ValueError("filenames and models both not set!")

        self.models = nn.ModuleList(models)

        for model_idx, model in enumerate(self.models):
            if self.models[0].transition_scheme() != model.transition_scheme():
                raise ValueError("Models {} and {} are incompatible.  {} vs {}".format(filenames[0], filenames[model_idx], self.models[0].transition_scheme(), model.transition_scheme()))
            if self.models[0].transitions != model.transitions:
                raise ValueError("Models %s and %s are incompatible: different transitions" % (filenames[0], filenames[model_idx]))
            if self.models[0].constituents != model.constituents:
                raise ValueError("Models %s and %s are incompatible: different constituents" % (filenames[0], filenames[model_idx]))
            if self.models[0].root_labels != model.root_labels:
                raise ValueError("Models %s and %s are incompatible: different root_labels" % (filenames[0], filenames[model_idx]))
            if self.models[0].uses_xpos() != model.uses_xpos():
                raise ValueError("Models %s and %s are incompatible: different uses_xpos" % (filenames[0], filenames[model_idx]))
            if self.models[0].reverse_sentence != model.reverse_sentence:
                raise ValueError("Models %s and %s are incompatible: different reverse_sentence" % (filenames[0], filenames[model_idx]))

        self._reverse_sentence = self.models[0].reverse_sentence


    @property
    def transitions(self):
        return self.models[0].transitions

    @property
    def root_labels(self):
        return self.models[0].root_labels

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def reverse_sentence(self):
        return self._reverse_sentence

    def uses_xpos(self):
        return self.models[0].uses_xpos()

    def log_norms(self):
        lines = ["NORMS FOR MODEL PARAMETERS"]
        for model_idx, model in enumerate(self.models):
            lines.append("  ---- MODEL %d ----" % model_idx)
            lines.extend(model.get_norms())
        logger.info("\n".join(lines))

    def log_shapes(self):
        lines = ["NORMS FOR MODEL PARAMETERS"]
        for name, param in self.named_parameters():
            if param.requires_grad:
                lines.append("{} {}".format(name, param.shape))
        logger.info("\n".join(lines))

    def get_params(self):
        return [x.get_params() for x in self.models]

    def build_batch_from_tagged_words(self, batch_size, data_iterator):
        """
        Read from the data_iterator batch_size tagged sentences and turn them into new parsing states

        Expects a list of list of (word, tag)
        """
        state_batch = []
        for _ in range(batch_size):
            sentence = next(data_iterator, None)
            if sentence is None:
                break
            state_batch.append(sentence)

        if len(state_batch) > 0:
            state_batch = [model.initial_state_from_words(state_batch) for model in self.models]
            state_batch = list(zip(*state_batch))
            state_batch = [MultiState(states, None, None, 0.0) for states in state_batch]
        return state_batch

    def build_batch_from_trees(self, batch_size, data_iterator):
        """
        Read from the data_iterator batch_size trees and turn them into N lists of parsing states
        """
        state_batch = []
        for _ in range(batch_size):
            gold_tree = next(data_iterator, None)
            if gold_tree is None:
                break
            state_batch.append(gold_tree)

        if len(state_batch) > 0:
            state_batch = [model.initial_state_from_gold_trees(state_batch) for model in self.models]
            state_batch = list(zip(*state_batch))
            state_batch = [MultiState(states, None, None, 0.0) for states in state_batch]
        return state_batch

    def predict(self, states, is_legal=True):
        states = list(zip(*[x.states for x in states]))
        predictions = [model.forward(state_batch) for model, state_batch in zip(self.models, states)]
        predictions = torch.stack(predictions)
        predictions = torch.sum(predictions, dim=0)

        model = self.models[0]

        # TODO: possibly refactor with lstm_model.predict
        pred_max = torch.argmax(predictions, dim=1)
        scores = torch.take_along_dim(predictions, pred_max.unsqueeze(1), dim=1)
        pred_max = pred_max.detach().cpu()

        pred_trans = [model.transitions[pred_max[idx]] for idx in range(len(states[0]))]
        if is_legal:
            for idx, (state, trans) in enumerate(zip(states[0], pred_trans)):
                if not trans.is_legal(state, model):
                    _, indices = predictions[idx, :].sort(descending=True)
                    for index in indices:
                        if model.transitions[index].is_legal(state, model):
                            pred_trans[idx] = model.transitions[index]
                            scores[idx] = predictions[idx, index]
                            break
                    else: # yeah, else on a for loop, deal with it
                        pred_trans[idx] = None
                        scores[idx] = None

        return predictions, pred_trans, scores.squeeze(1)

    def bulk_apply(self, state_batch, transitions, fail=False):
        new_states = []

        states = list(zip(*[x.states for x in state_batch]))
        states = [x.bulk_apply(y, transitions, fail=fail) for x, y in zip(self.models, states)]
        states = list(zip(*states))
        state_batch = [x._replace(states=y) for x, y in zip(state_batch, states)]
        return state_batch

    def parse_sentences(self, data_iterator, build_batch_fn, batch_size, transition_choice, keep_state=False, keep_constituents=False, keep_scores=False):
        """
        Repeat transitions to build a list of trees from the input batches.

        The data_iterator should be anything which returns the data for a parse task via next()
        build_batch_fn is a function that turns that data into State objects
        This will be called to generate batches of size batch_size until the data is exhausted

        The return is a list of tuples: (gold_tree, [(predicted, score) ...])
        gold_tree will be left blank if the data did not include gold trees
        currently score is always 1.0, but the interface may be expanded
        to get a score from the result of the parsing

        transition_choice: which method of the model to use for
        choosing the next transition

        TODO: refactor with base_model
        """
        treebank = []
        treebank_indices = []
        # this will produce tuples of states
        # batch size lists of num models tuples
        state_batch = build_batch_fn(batch_size, data_iterator)
        batch_indices = list(range(len(state_batch)))
        horizon_iterator = iter([])

        if keep_constituents:
            constituents = defaultdict(list)

        while len(state_batch) > 0:
            pred_scores, transitions, scores = transition_choice(state_batch)
            # num models lists of batch size states
            state_batch = self.bulk_apply(state_batch, transitions)

            remove = set()
            for idx, states in enumerate(state_batch):
                if states.finished(self):
                    predicted_tree = states.get_tree(self)
                    if self.reverse_sentence:
                        predicted_tree = predicted_tree.reverse()
                    gold_tree = states.gold_tree
                    # TODO: could easily store the score here
                    # not sure what it means to store the state,
                    # since each model is tracking its own state
                    treebank.append(ParseResult(gold_tree, [ScoredTree(predicted_tree, None)], None, None))
                    treebank_indices.append(batch_indices[idx])
                    remove.add(idx)

            if len(remove) > 0:
                state_batch = [state for idx, state in enumerate(state_batch) if idx not in remove]
                batch_indices = [batch_idx for idx, batch_idx in enumerate(batch_indices) if idx not in remove]

            for _ in range(batch_size - len(state_batch)):
                horizon_state = next(horizon_iterator, None)
                if not horizon_state:
                    horizon_batch = build_batch_fn(batch_size, data_iterator)
                    if len(horizon_batch) == 0:
                        break
                    horizon_iterator = iter(horizon_batch)
                    horizon_state = next(horizon_iterator, None)

                state_batch.append(horizon_state)
                batch_indices.append(len(treebank) + len(state_batch))

        treebank = utils.unsort(treebank, treebank_indices)
        return treebank

    def parse_sentences_no_grad(self, data_iterator, build_batch_fn, batch_size, transition_choice, keep_state=False, keep_constituents=False, keep_scores=False):
        with torch.no_grad():
            return self.parse_sentences(data_iterator, build_batch_fn, batch_size, transition_choice, keep_state, keep_constituents, keep_scores)

class EnsembleTrainer(BaseTrainer):
    """
    Stores a list of constituency models, useful for combining their results into one stronger model
    """
    def __init__(self, ensemble, optimizer=None, scheduler=None, epochs_trained=0, batches_trained=0, best_f1=0.0, best_epoch=0):
        super().__init__(ensemble, optimizer, scheduler, epochs_trained, batches_trained, best_f1, best_epoch)

    @staticmethod
    def from_files(args, filenames, foundation_cache=None):
        ensemble = Ensemble(args, filenames, foundation_cache=foundation_cache)
        ensemble = ensemble.to(args.get('device', None))
        return EnsembleTrainer(ensemble)

    def get_peft_params(self):
        params = []
        for model in self.model.models:
            if model.args.get('use_peft', False):
                from peft import get_peft_model_state_dict
                params.append(get_peft_model_state_dict(model.bert_model, adapter_name=model.peft_name))
            else:
                params.append(None)

        return params

    @property
    def model_type(self):
        return ModelType.ENSEMBLE

    @staticmethod
    def model_from_params(params, peft_params, args, foundation_cache=None, peft_name=None):
        # TODO: fill in peft_name
        if peft_params is None:
            peft_params = [None] * len(params)
        if peft_name is None:
            peft_name = [None] * len(params)

        if len(params) != len(peft_params):
            raise ValueError("Model file had params length %d and peft params length %d" % (len(params), len(peft_params)))
        if len(params) != len(peft_name):
            raise ValueError("Model file had params length %d and peft name length %d" % (len(params), len(peft_name)))

        models = [Trainer.model_from_params(model_param, peft_param, args, foundation_cache, peft_name=pname)
                  for model_param, peft_param, pname in zip(params, peft_params, peft_name)]
        ensemble = Ensemble(args, models=models)
        ensemble = ensemble.to(args.get('device', None))
        return ensemble

DEFAULT_EVAL = {
    "en": "en_wsj_dev.mrg",
    "id": "id_icon_dev.mrg",
    "vi": "vi_vlsp22_dev.mrg",
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--tokenized_file', type=str, default=None, help='Input file of tokenized text for parsing with parse_text.')
    parser.add_argument('--tokenized_dir', type=str, default=None, help='Input directory of tokenized text for parsing with parse_text.')

    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')

    utils.add_device_args(parser)

    parser.add_argument('--lang', default='en', help='Language to use')

    parser.add_argument('--eval_batch_size', type=int, default=50, help='How many trees to batch when running eval')
    parser.add_argument('models', type=str, nargs='+', default=None, help="Which model(s) to load")

    parser.add_argument('--mode', default='predict', choices=['parse_text', 'predict'])
    add_predict_output_args(parser)

    retagging.add_retag_args(parser)

    args = vars(parser.parse_args())

    retagging.postprocess_args(args)
    postprocess_predict_output_args(args)
    args['num_generate'] = 0

    if not args['eval_file'] and args['lang'] in DEFAULT_EVAL:
        paths = get_default_paths()
        args['eval_file'] = os.path.join(paths["CONSTITUENCY_DATA_DIR"], DEFAULT_EVAL[args['lang']])

    return args

def main():
    args = parse_args()
    utils.log_training_args(args, logger, name="ensemble")
    retag_pipeline = retagging.build_retag_pipeline(args)
    foundation_cache = retag_pipeline[0].foundation_cache if retag_pipeline else FoundationCache()

    ensemble = Ensemble(args, args['models'], foundation_cache)
    ensemble.eval()

    if args['mode'] == 'predict':
        with EvaluateParser() as evaluator:
            treebank = tree_reader.read_treebank(args['eval_file'])
            logger.info("Read %d trees for evaluation", len(treebank))

            if retag_pipeline is not None:
                logger.info("Retagging trees using the %s tags from the %s package...", args['retag_method'], args['retag_package'])
                retagged_treebank = retag_trees(treebank, retag_pipeline, args['retag_xpos'])
                logger.info("Retagging finished")

            f1, kbestF1, _ = run_dev_set(ensemble, retagged_treebank, treebank, args, evaluator)
            logger.info("F1 score on %s: %f", args['eval_file'], f1)
            if kbestF1 is not None:
                logger.info("KBest F1 score on %s: %f", args['eval_file'], kbestF1)
    elif args['mode'] == 'parse_text':
        if args['tokenized_dir']:
            if not args['predict_dir']:
                raise ValueError("Must specific --predict_dir to go with --tokenized_dir")
            parse_dir(args, ensemble, retag_pipeline, args['tokenized_dir'], args['predict_dir'])
        else:
            parse_text(args, ensemble, retag_pipeline)
    else:
        raise ValueError("Unhandled mode %s" % args['mode'])


if __name__ == "__main__":
    main()
