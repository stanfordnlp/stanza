"""
Build an ensemble of GraphParsers
"""

import argparse
import copy
from stanza.models.common import pretrain
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.depparse.model import GraphParser, EnsembleGraphParser
from stanza.models.depparse.transition.model import TransitionParser, EnsembleTransitionParser
from stanza.models.depparse.trainer import Trainer

def build_ensemble(args, pt, model_names, foundation_cache, device=None):
    args = copy.deepcopy(args)
    models = []
    trainers = []
    for load_name in model_names:
        tr = Trainer.load(load_name, pt, args=args, device=device, foundation_cache=foundation_cache)
        models.append(tr.model)
        trainers.append(tr)

    if all(isinstance(x, GraphParser) for x in models):
        ensemble = EnsembleGraphParser(args, models[0].vocab, models)
    elif all(isinstance(x, TransitionParser) for x in models):
        ensemble = EnsembleTransitionParser(args, models[0].vocab, models)
    else:
        raise ValueError("Not all models are an ensemble type!  %s" % ([type(x) for x in models]))

    if all(x.args.get('reversed', False) for x in models):
        args['reversed'] = True
    elif all(not x.args.get('reversed', False) for x in models):
        args['reversed'] = False
    else:
        raise ValueError("Models do not agree on data diredction!  %s" % ([x.args.get('reversed', False) for x in models]))

    tr = Trainer(args=args, vocab=ensemble.vocab, model=ensemble, build_optimizer=False)
    tr.model_name = ",".join(model_names)
    return tr, trainers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, default="graph_ensemble.pt", help="File name to save the model")
    parser.add_argument('load_name', nargs="+", type=str, help="Which models to use for the ensemble")

    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')

    args = vars(parser.parse_args())

    pt = None
    if args['wordvec_pretrain_file']:
        pt = pretrain.Pretrain(args['wordvec_pretrain_file'])
        args['pretrain'] = True

    foundation_cache = FoundationCache()

    tr, _ = build_ensemble(args, pt, args['load_name'], foundation_cache)
    tr.save(args['save_name'])

if __name__ == '__main__':
    main()
