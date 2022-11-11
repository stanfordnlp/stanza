"""
Refactor a few functions specifically for retagging trees

Retagging is important because the gold tags will not be available at runtime

Note that the method which does the actual retagging is in utils.py
so as to avoid unnecessary circular imports
(eg, Pipeline imports constituency/trainer which imports this which imports Pipeline)
"""

import logging

from stanza import Pipeline

from stanza.models.common.vocab import VOCAB_PREFIX

logger = logging.getLogger('stanza')

def add_retag_args(parser):
    """
    Arguments specifically for retagging treebanks
    """
    parser.add_argument('--retag_package', default="default", help='Which tagger shortname to use when retagging trees.  None for no retagging.  Retagging is recommended, as gold tags will not be available at pipeline time')
    parser.add_argument('--retag_method', default='xpos', choices=['xpos', 'upos'], help='Which tags to use when retagging')
    parser.add_argument('--retag_model_path', default=None, help='Path to a retag POS model to use.  Will use a downloaded Stanza model by default')
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
    Build a retag pipeline based on the arguments

    May alter the arguments if the pipeline is incompatible, such as
    taggers with no xpos
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
        retag_args = {"lang": lang,
                      "processors": "tokenize, pos",
                      "tokenize_pretokenized": True,
                      "package": {"pos": package}}
        if args['retag_model_path'] is not None:
            retag_args['pos_model_path'] = args['retag_model_path']
        retag_pipeline = Pipeline(**retag_args)
        if args['retag_xpos'] and len(retag_pipeline.processors['pos'].vocab['xpos']) == len(VOCAB_PREFIX):
            logger.warning("XPOS for the %s tagger is empty.  Switching to UPOS", package)
            args['retag_xpos'] = False
            args['retag_method'] = 'upos'
        return retag_pipeline

    return None
