"""
basic demo script
"""

import argparse
import os

from pathlib import Path
from stanfordnlp import Document, Pipeline
from stanfordnlp.utils.resources import build_default_config


if __name__ == '__main__':
    # get arguments
    # determine home directory
    home_dir = str(Path.home())
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--models_dir', help='location of models files | default: ~/stanfordnlp_data', 
                        default=home_dir+'/stanfordnlp_data')
    args = parser.parse_args()
    # download the models
    if not os.path.exists(args.models_dir+'/en_ewt_models'):
        download('en_ewt')
    # set up a pipeline
    print('---')
    print('Building pipeline...')
    print('with config: ')
    pipeline_config = build_default_config('en_ewt', args.models_dir)
    print(pipeline_config)
    print('')
    pipeline = Pipeline(config=pipeline_config)
    # set up document
    doc = Document('Barack Obama was born in Hawaii.  He was elected president in 2008.')
    # run pipeline on the document
    pipeline.process(doc)
    # access nlp annotations
    print('')
    print('---')
    print('tokens of first sentence: ')
    for tok in doc.sentences[0].tokens:
        print(tok.word + '\t' + tok.lemma + '\t' + tok.pos)
    print('')
    print('---')
    print('dependency parse of first sentence: ')
    for dep_edge in doc.sentences[0].dependencies:
        print((dep_edge[0].word, dep_edge[1], dep_edge[2].word))
    print('')

