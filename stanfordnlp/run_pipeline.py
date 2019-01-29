"""
script for running full pipeline on command line
"""

import argparse
import os

from stanfordnlp import download, Pipeline
from stanfordnlp.utils.resources import default_treebanks, DEFAULT_MODEL_DIR


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--models_dir', help='location of models files | default: ~/stanfordnlp_resources', 
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument('-l', '--language', help='language of text | default: en', default='en')
    parser.add_argument('-p', '--processors', help='list of processors to run | default: "tokenize,mwt,pos,lemma,depparse"',
                        default='tokenize,mwt,pos,lemma,depparse')
    parser.add_argument('text_file')
    args = parser.parse_args()
    # set output file path
    output_file_path = args.text_file+'.out'
    # map language code to treebank shorthand
    treebank_shorthand = default_treebanks[args.language]
    # check for models
    print('checking for models...')
    lang_models_dir = '%s/%s_models' % (args.models_dir, treebank_shorthand)
    if not os.path.exists(lang_models_dir):
        print('could not find: '+lang_models_dir)
        download(args.language, resource_dir=args.models_dir)
    # set up pipeline
    pipeline = Pipeline(processors=args.processors,lang=args.language, models_dir=args.models_dir)
    # build document
    print('running pipeline...')
    doc = pipeline(open(args.text_file).read())
    # write conll to file
    doc.write_conll_to_file(output_file_path)
    print('done.')
    print('results written to: '+output_file_path)

