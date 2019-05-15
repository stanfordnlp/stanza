"""
script for running full pipeline on command line
"""

import argparse
import os

from stanfordnlp import download, Pipeline
from stanfordnlp.pipeline.core import BOOLEAN_PROCESSOR_SETTINGS_LIST, PROCESSOR_SETTINGS_LIST
from stanfordnlp.utils.resources import default_treebanks, DEFAULT_MODEL_DIR


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    # main arguments
    parser.add_argument('-d', '--models-dir', help='location of models files | default: ~/stanfordnlp_resources',
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument('-l', '--language', help='language of text | default: en', default='en')
    parser.add_argument('-t', '--treebank', help='treebank to use | default: None', default=None)
    parser.add_argument('-o', '--output', help='output file path', default=None)
    parser.add_argument('-p', '--processors',
                        help='list of processors to run | default: "tokenize,mwt,pos,lemma,depparse"',
                        default='tokenize,mwt,pos,lemma,depparse')
    # misc arguments
    parser.add_argument('--force-download', help='force download of models', action='store_true')
    # processor related arguments
    for processor_setting in PROCESSOR_SETTINGS_LIST:
        if processor_setting in BOOLEAN_PROCESSOR_SETTINGS_LIST:
            parser.add_argument('--' + processor_setting, action='store_true', default=None, help=argparse.SUPPRESS)
        else:
            parser.add_argument('--' + processor_setting, help=argparse.SUPPRESS)
    parser.add_argument('text_file')
    args = parser.parse_args()
    # set output file path
    if args.output is None:
        output_file_path = args.text_file+'.out'
    else:
        output_file_path = args.output
    # map language code to treebank shorthand
    if args.treebank is not None:
        treebank_shorthand = args.treebank
    else:
        treebank_shorthand = default_treebanks[args.language]
    # check for models
    print('checking for models...')
    lang_models_dir = '%s/%s_models' % (args.models_dir, treebank_shorthand)
    if not os.path.exists(lang_models_dir):
        print('could not find: '+lang_models_dir)
        download(treebank_shorthand, resource_dir=args.models_dir, force=args.force_download)
    # set up pipeline
    pipeline_config = \
        dict([(k, v) for k, v in vars(args).items() if k in PROCESSOR_SETTINGS_LIST and v is not None])
    pipeline = Pipeline(processors=args.processors, treebank=treebank_shorthand, models_dir=args.models_dir, **pipeline_config)
    # build document
    print('running pipeline...')
    doc = pipeline(open(args.text_file).read())
    # write conll to file
    doc.write_conll_to_file(output_file_path)
    print('done.')
    print('results written to: '+output_file_path)

