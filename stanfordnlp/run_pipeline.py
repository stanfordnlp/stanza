"""
script for running full pipeline on command line
"""

import argparse
import os

from stanfordnlp import download, Document, Pipeline
from stanfordnlp.utils.resources import default_treebanks, DEFAULT_MODEL_DIR


if __name__ == '__main__':
    # get arguments
    # determine home directory
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--models_dir', help='location of models files | default: ~/stanfordnlp_data', 
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument('-l', '--language', help='language of text | default: en_ewt', default='en_ewt')
    parser.add_argument('text_file')
    args = parser.parse_args()
    output_file_path = args.text_file+'.out'
    treebank_shorthand = default_treebanks[args.language]
    # check for models
    print('checking for models...')
    lang_models_dir = '%s/%s_models' % (args.models_dir, treebank_shorthand)
    if not os.path.exists(lang_models_dir):
        print('could not find: '+lang_models_dir)
        download(args.language, resource_dir=args.models_dir)
    # set up pipeline
    pipeline = Pipeline(lang=args.language, models_dir=args.models_dir)
    # run process
    # load input text
    input_text = open(args.text_file).read()
    # build document
    print('running pipeline...')
    doc = pipeline(input_text)
    # write conll to file
    doc.write_conll_to_file(output_file_path)
    print('done.')
    print('results written to: '+output_file_path)

