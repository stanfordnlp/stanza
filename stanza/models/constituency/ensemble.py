"""
Prototype of ensembling N models together on the same dataset

The main process is to run the normal transition sequence, but sum the
scores for the N models and use that to choose the highest scoring
transition

Currently the code is kind of awkward because it includes a lot of
duplicated logic in predict() and parse_sentences()
"""


import argparse
import logging

import torch

from stanza.models.common import utils
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import retagging
from stanza.models.constituency import tree_reader
from stanza.models.constituency.trainer import Trainer, run_dev_set
from stanza.models.constituency.utils import retag_trees
from stanza.server.parser_eval import EvaluateParser, ParseResult, ScoredTree

logger = logging.getLogger('stanza.constituency.trainer')

class Ensemble:
    def __init__(self, filenames, args, foundation_cache=None):
        """
        Loads each model in filenames

        If foundation_cache is None, we build one on our own,
        as the expectation is the models will reuse modules
        such as pretrain, charlm, bert
        """
        if foundation_cache is None:
            foundation_cache = FoundationCache()

        if isinstance(filenames, str):
            filenames = [filenames]
        self.models = [Trainer.load(filename, args, load_optimizer=False, foundation_cache=foundation_cache).model for filename in filenames]

        for model_idx, model in enumerate(self.models):
            if self.models[0].transition_scheme() != model.transition_scheme():
                raise ValueError("Models {} and {} are incompatible.  {} vs {}".format(filenames[0], filenames[model_idx], self.models[0].transition_scheme(), model.transition_scheme()))
            if self.models[0].transitions != model.transitions:
                raise ValueError("Models %s and %s are incompatible: different transitions" % (filenames[0], filenames[model_idx]))
            if self.models[0].constituents != model.constituents:
                raise ValueError("Models %s and %s are incompatible: different constituents" % (filenames[0], filenames[model_idx]))
            if self.models[0].root_labels != model.root_labels:
                raise ValueError("Models %s and %s are incompatible: different root_labels" % (filenames[0], filenames[model_idx]))

    def eval(self):
        for model in self.models:
            model.eval()

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
        return state_batch

    def predict(self, states, is_legal=True):
        states = list(zip(*states))
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
            #print(batch_indices)
            pred_scores, transitions, scores = transition_choice(state_batch)
            # num models lists of batch size states
            state_batch = list(zip(*state_batch))
            state_batch = [parse_transitions.bulk_apply(model, states, transitions) for model, states in zip(self.models, state_batch)]

            remove = set()
            for idx, state in enumerate(state_batch[0]):
                if state.finished(self.models[0]):
                    predicted_tree = state.get_tree(self.models[0])
                    gold_tree = state.gold_tree
                    # TODO: could easily store the score here
                    # not sure what it means to store the state,
                    # since each model is tracking its own state
                    treebank.append(ParseResult(gold_tree, [ScoredTree(predicted_tree, None)], None, None))
                    treebank_indices.append(batch_indices[idx])
                    remove.add(idx)

            # batch size lists of num models tuples
            state_batch = list(zip(*state_batch))

            if len(remove) > 0:
                # remove a whole tuple of states at once
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

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval_file', type=str, default="data/constituency/en_wsj_dev.mrg", help='Input file for data loader.')
    parser.add_argument('--charlm_forward_file', type=str, default="/home/john/stanza_resources/en/forward_charlm/1billion.pt", help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default="/home/john/stanza_resources/en/backward_charlm/1billion.pt", help="Exact path to use for backward charlm")
    parser.add_argument('--wordvec_pretrain_file', type=str, default="/home/john/stanza_resources/en/pretrain/combined.pt", help='Exact name of the pretrain file to read')

    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    parser.add_argument('--lang', default='en', help='Language to use')

    parser.add_argument('--eval_batch_size', type=int, default=50, help='How many trees to batch when running eval')
    parser.add_argument('models', type=str, nargs='+', default=None, help="Which model(s) to load")

    retagging.add_retag_args(parser)
    parser.set_defaults(retag_method='xpos')

    args = vars(parser.parse_args())

    retagging.postprocess_args(args)
    args['num_generate'] = 0
    args['mode'] = 'predict'
    args['predict_file'] = None
    args['predict_dir'] = None

    return args

def main():
    args = parse_args()
    retag_pipeline = retagging.build_retag_pipeline(args)
    foundation_cache = retag_pipeline.foundation_cache if retag_pipeline else FoundationCache()

    ensemble = Ensemble(args['models'], args, foundation_cache)

    with EvaluateParser() as evaluator:
        treebank = tree_reader.read_treebank(args['eval_file'])
        logger.info("Read %d trees for evaluation", len(treebank))

        if retag_pipeline is not None:
            logger.info("Retagging trees using the %s tags from the %s package...", args['retag_method'], args['retag_package'])
            treebank = retag_trees(treebank, retag_pipeline, args['retag_xpos'])
            logger.info("Retagging finished")

        f1, kbestF1 = run_dev_set(ensemble, treebank, args, evaluator)
        logger.info("F1 score on %s: %f", args['eval_file'], f1)
        if kbestF1 is not None:
            logger.info("KBest F1 score on %s: %f", args['eval_file'], kbestF1)


if __name__ == "__main__":
    main()
