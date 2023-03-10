"""
Refactor a few functions specifically for retagging trees

Retagging is important because the gold tags will not be available at runtime

Note that the method which does the actual retagging is in utils.py
so as to avoid unnecessary circular imports
(eg, Pipeline imports constituency/trainer which imports this which imports Pipeline)
"""

import copy
import logging

from stanza import Pipeline

from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.common.vocab import VOCAB_PREFIX

logger = logging.getLogger('stanza')

def add_retag_args(parser):
    """
    Arguments specifically for retagging treebanks
    """
    parser.add_argument('--retag_package', default="default", help='Which tagger shortname to use when retagging trees.  None for no retagging.  Retagging is recommended, as gold tags will not be available at pipeline time')
    parser.add_argument('--retag_method', default='xpos', choices=['xpos', 'upos'], help='Which tags to use when retagging')
    parser.add_argument('--retag_model_path', default=None, help='Path to a retag POS model to use.  Will use a downloaded Stanza model by default.  Can specify multiple taggers with ; in which case the majority vote wins')
    parser.add_argument('--retag_pretrain_path', default=None, help='Use this for a pretrain path for the retagging pipeline.  Generally not needed unless using a custom POS model with a custom pretrain')
    parser.add_argument('--retag_charlm_forward_file', default=None, help='Use this for a forward charlm path for the retagging pipeline.  Generally not needed unless using a custom POS model with a custom charlm')
    parser.add_argument('--retag_charlm_backward_file', default=None, help='Use this for a backward charlm  path for the retagging pipeline.  Generally not needed unless using a custom POS model with a custom charlm')
    parser.add_argument('--no_retag', dest='retag_package', action="store_const", const=None, help="Don't retag the trees")

def postprocess_args(args):
    """
    After parsing args, unify some settings
    """
    if args['retag_method'] == 'xpos':
        args['retag_xpos'] = True
    elif args['retag_method'] == 'upos':
        args['retag_xpos'] = False
    else:
        raise ValueError("Unknown retag method {}".format(xpos))

def build_retag_pipeline(args):
    """
    Builds retag pipelines based on the arguments

    May alter the arguments if the pipeline is incompatible, such as
    taggers with no xpos

    Will return a list of one or more retag pipelines.
    Multiple tagger models can be specified by having them
    semi-colon separated in retag_model_path.
    """
    # some argument sets might not use 'mode'
    if args['retag_package'] is not None and args.get('mode', None) != 'remove_optimizer':
        if '_' in args['retag_package']:
            lang, package = args['retag_package'].split('_', 1)
        else:
            if 'lang' not in args:
                raise ValueError("Retag package %s does not specify the language, and it is not clear from the arguments" % args['retag_package'])
            lang = args.get('lang', None)
            package = args['retag_package']
        foundation_cache = FoundationCache()
        retag_args = {"lang": lang,
                      "processors": "tokenize, pos",
                      "tokenize_pretokenized": True,
                      "package": {"pos": package}}
        if args['retag_pretrain_path'] is not None:
            retag_args['pos_pretrain_path'] = args['retag_pretrain_path']
        if args['retag_charlm_forward_file'] is not None:
            retag_args['pos_forward_charlm_path'] = args['retag_charlm_forward_file']
        if args['retag_charlm_backward_file'] is not None:
            retag_args['pos_backward_charlm_path'] = args['retag_charlm_backward_file']

        def build(retag_args, path):
            retag_args = copy.deepcopy(retag_args)
            if path is not None:
                retag_args['allow_unknown_language'] = True
                retag_args['pos_model_path'] = path

            retag_pipeline = Pipeline(foundation_cache=foundation_cache, **retag_args)
            if args['retag_xpos'] and len(retag_pipeline.processors['pos'].vocab['xpos']) == len(VOCAB_PREFIX):
                logger.warning("XPOS for the %s tagger is empty.  Switching to UPOS", package)
                args['retag_xpos'] = False
                args['retag_method'] = 'upos'
            return retag_pipeline

        if args['retag_model_path'] is None:
            return [build(retag_args, None)]
        paths = args['retag_model_path'].split(";")
        # can be length 1 if only one tagger to work with
        return [build(retag_args, path) for path in paths]

    return None
