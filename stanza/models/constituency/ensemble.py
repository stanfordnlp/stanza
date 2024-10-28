"""
Prototype of ensembling N models together on the same dataset

The main inference method is to run the normal transition sequence,
but sum the scores for the N models and use that to choose the highest
scoring transition

Example of how to run it to build a silver dataset
(or just parse a text file in general):

# first, use this tool to build a saved ensemble
python3 stanza/models/constituency/ensemble.py
   saved_models/constituency/wsj_inorder_?.pt
   --save_name saved_models/constituency/en_ensemble.pt

# then use the ensemble directly as a model in constituency_parser.py
python3 stanza/models/constituency_parser.py
   --save_name saved_models/constituency/en_ensemble.pt
   --mode parse_text
   --tokenized_file /nlp/scr/horatio/en_silver/en_split_100
   --predict_file /nlp/scr/horatio/en_silver/en_split_100.inorder.mrg
   --retag_package en_combined_bert
   --lang en

then, ideally, run a second time with a set of topdown models,
then take the trees which match from the files
"""


import argparse
import copy
import logging
import os

import torch
import torch.nn as nn

from stanza.models.common import utils
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.constituency.base_trainer import BaseTrainer, ModelType
from stanza.models.constituency.state import MultiState
from stanza.models.constituency.trainer import Trainer
from stanza.models.constituency.utils import build_optimizer, build_scheduler
from stanza.server.parser_eval import ParseResult, ScoredTree

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

        self.args = args
        if filenames:
            if models:
                raise ValueError("both filenames and models set when making the Ensemble")

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
                raise ValueError(f"Models {filenames[0]} and {filenames[model_idx]} are incompatible: different transitions\n{filenames[0]}:\n{self.models[0].transitions}\n{filenames[model_idx]}:\n{model.transitions}")
            if self.models[0].constituents != model.constituents:
                raise ValueError("Models %s and %s are incompatible: different constituents" % (filenames[0], filenames[model_idx]))
            if self.models[0].root_labels != model.root_labels:
                raise ValueError("Models %s and %s are incompatible: different root_labels" % (filenames[0], filenames[model_idx]))
            if self.models[0].uses_xpos() != model.uses_xpos():
                raise ValueError("Models %s and %s are incompatible: different uses_xpos" % (filenames[0], filenames[model_idx]))
            if self.models[0].reverse_sentence != model.reverse_sentence:
                raise ValueError("Models %s and %s are incompatible: different reverse_sentence" % (filenames[0], filenames[model_idx]))

        self._reverse_sentence = self.models[0].reverse_sentence

        # submodels are not trained (so far)
        self.detach_submodels()

        logger.debug("Number of models in the Ensemble: %d", len(self.models))
        self.register_parameter('weighted_sum', torch.nn.Parameter(torch.zeros(len(self.models), len(self.transitions), requires_grad=True)))

    def detach_submodels(self):
        # submodels are not trained (so far)
        for model in self.models:
            for _, parameter in model.named_parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode:
            # peft has a weird interaction where it turns requires_grad back on
            # even if it was previously off
            self.detach_submodels()

    @property
    def transitions(self):
        return self.models[0].transitions

    @property
    def root_labels(self):
        return self.models[0].root_labels

    @property
    def device(self):
        return next(self.parameters()).device

    def unary_limit(self):
        """
        Limit on the number of consecutive unary transitions
        """
        return min(m.unary_limit() for m in self.models)

    def transition_scheme(self):
        return self.models[0].transition_scheme()

    def has_unary_transitions(self):
        return self.models[0].has_unary_transitions()

    @property
    def is_top_down(self):
        return self.models[0].is_top_down

    @property
    def reverse_sentence(self):
        return self._reverse_sentence

    @property
    def retag_method(self):
        # TODO: make the method an enum
        return self.models[0].args['retag_method']

    def uses_xpos(self):
        return self.models[0].uses_xpos()

    def get_top_constituent(self, constituents):
        return self.models[0].get_top_constituent(constituents)

    def get_top_transition(self, transitions):
        return self.models[0].get_top_transition(transitions)

    def log_norms(self):
        lines = ["NORMS FOR MODEL PARAMETERS"]
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("models."):
                zeros = torch.sum(param.abs() < 0.000001).item()
                norm = "%.6g" % torch.norm(param).item()
                lines.append("%s %s %d %d" % (name, norm, zeros, param.nelement()))
        for model_idx, model in enumerate(self.models):
            sublines = model.get_norms()
            if len(sublines) > 0:
                lines.append("  ---- MODEL %d ----" % model_idx)
                lines.extend(sublines)
        logger.info("\n".join(lines))

    def log_shapes(self):
        lines = ["NORMS FOR MODEL PARAMETERS"]
        for name, param in self.named_parameters():
            if param.requires_grad:
                lines.append("{} {}".format(name, param.shape))
        logger.info("\n".join(lines))

    def get_params(self):
        model_state = self.state_dict()
        # don't save the children in the base params
        model_state = {k: v for k, v in model_state.items() if not k.startswith("models.")}
        return {
            "base_params": model_state,
            "children_params": [x.get_params() for x in self.models]
        }

    def initial_state_from_preterminals(self, preterminal_lists, gold_trees, gold_sequences):
        state_batch = [model.initial_state_from_preterminals(preterminal_lists, gold_trees, gold_sequences) for model in self.models]
        state_batch = list(zip(*state_batch))
        state_batch = [MultiState(states, gold_tree, gold_sequence, 0.0)
                       for states, gold_tree, gold_sequence in zip(state_batch, gold_trees, gold_sequences)]
        return state_batch

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

        # batch X num transitions X num models
        predictions = torch.stack(predictions, dim=2)

        flat_predictions = torch.einsum("BTM,MT->BT", predictions, self.weighted_sum)
        predictions = torch.sum(predictions, dim=2) + flat_predictions

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

    def parse_tagged_words(self, words, batch_size):
        """
        This parses tagged words and returns a list of trees.

        `parse_tagged_words` is useful at Pipeline time -
          it takes words & tags and processes that into trees.

        The tagged words should be represented:
          one list per sentence
            each sentence is a list of (word, tag)
        The return value is a list of ParseTree objects

        TODO: this really ought to be refactored with base_model
        """
        logger.debug("Processing %d sentences", len(words))
        self.eval()

        sentence_iterator = iter(words)
        treebank = self.parse_sentences_no_grad(sentence_iterator, self.build_batch_from_tagged_words, batch_size, self.predict, keep_state=False, keep_constituents=False)

        results = [t.predictions[0].tree for t in treebank]
        return results

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
    def __init__(self, ensemble, optimizer=None, scheduler=None, epochs_trained=0, batches_trained=0, best_f1=0.0, best_epoch=0, first_optimizer=False):
        super().__init__(ensemble, optimizer, scheduler, epochs_trained, batches_trained, best_f1, best_epoch, first_optimizer)

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

    def log_num_words_known(self, words):
        nwk = [m.num_words_known(words) for m in self.model.models]
        if all(x == nwk[0] for x in nwk):
            logger.info("Number of words in the training set known to each sub-model: %d out of %d", nwk[0], len(words))
        else:
            logger.info("Number of words in the training set known to the sub-models:\n  %s" % "\n  ".join(["%d/%d" % (x, len(words)) for x in nwk]))

    @staticmethod
    def build_optimizer(args, model, first_optimizer):
        def fake_named_parameters():
            for n, p in model.named_parameters():
                if not n.startswith("models."):
                    yield n, p

        # TODO: there has to be a cleaner way to do this, like maybe a "keep" callback
        # TODO: if we finetune the underlying models, we will want a series of optimizers
        # so that they can have a different learning rate from the ensemble's fields
        fake_model = copy.copy(model)
        fake_model.named_parameters = fake_named_parameters
        optimizer = build_optimizer(args, fake_model, first_optimizer)
        return optimizer

    @staticmethod
    def load_optimizer(model, checkpoint, first_optimizer, filename):
        optimizer = EnsembleTrainer.build_optimizer(model.models[0].args, model, first_optimizer)
        if checkpoint.get('optimizer_state_dict', None) is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                raise ValueError("Failed to load optimizer from %s" % filename) from e
        else:
            logger.info("Attempted to load optimizer to resume training, but optimizer not saved.  Creating new optimizer")
        return optimizer

    @staticmethod
    def load_scheduler(model, optimizer, checkpoint, first_optimizer):
        scheduler = build_scheduler(model.models[0].args, optimizer, first_optimizer=first_optimizer)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return scheduler

    @staticmethod
    def model_from_params(params, peft_params, args, foundation_cache=None, peft_name=None):
        # TODO: no need for the if/else once the models are rebuilt
        children_params = params["children_params"] if isinstance(params, dict) else params
        base_params = params["base_params"] if isinstance(params, dict) else {}

        # TODO: fill in peft_name
        if peft_params is None:
            peft_params = [None] * len(children_params)
        if peft_name is None:
            peft_name = [None] * len(children_params)

        if len(children_params) != len(peft_params):
            raise ValueError("Model file had params length %d and peft params length %d" % (len(params), len(peft_params)))
        if len(children_params) != len(peft_name):
            raise ValueError("Model file had params length %d and peft name length %d" % (len(params), len(peft_name)))

        models = [Trainer.model_from_params(model_param, peft_param, args, foundation_cache, peft_name=pname)
                  for model_param, peft_param, pname in zip(children_params, peft_params, peft_name)]
        ensemble = Ensemble(args, models=models)
        ensemble.load_state_dict(base_params, strict=False)
        ensemble = ensemble.to(args.get('device', None))
        return ensemble

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')

    utils.add_device_args(parser)

    parser.add_argument('--lang', default='en', help='Language to use')

    parser.add_argument('models', type=str, nargs='+', default=None, help="Which model(s) to load")

    parser.add_argument('--save_name', type=str, default=None, required=True, help='Where to save the combined ensemble')

    args = vars(parser.parse_args())

    return args

def main(args=None):
    args = parse_args(args)
    foundation_cache = FoundationCache()

    ensemble = EnsembleTrainer.from_files(args, args['models'], foundation_cache)
    ensemble.save(args['save_name'], save_optimizer=False)

if __name__ == "__main__":
    main()
