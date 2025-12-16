"""
Build an ensemble of GraphParsers
"""

import argparse
from stanza.models.common import pretrain
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.depparse.model import GraphParser, EnsembleGraphParser
from stanza.models.depparse.transition.model import TransitionParser, EnsembleTransitionParser
from stanza.models.depparse.trainer import Trainer

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

    models = []
    for load_name in args['load_name']:
        tr = Trainer.load(load_name, pt, args=args, foundation_cache=foundation_cache)
        models.append(tr.model)

    # TODO:
    # check the vocabs are the same

    if all(isinstance(x, GraphParser) for x in models):
        ensemble = EnsembleGraphParser(args, models[0].vocab, models)
    elif all(isinstance(x, TransitionParser) for x in models):
        ensemble = EnsembleTransitionParser(args, models[0].vocab, models)
    else:
        raise ValueError("Not all models are an ensemble type!  %s" % ([type(x) for x in models]))
    tr = Trainer(args=args, vocab=ensemble.vocab, model=ensemble, build_optimizer=False)
    tr.save(args['save_name'])

if __name__ == '__main__':
    main()
