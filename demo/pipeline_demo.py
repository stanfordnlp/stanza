"""
basic demo script
"""

import argparse
import os

import stanfordnlp
from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--models_dir', help='location of models files | default: ~/stanfordnlp_data',
                        default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    # download the models
    demo_language = "en"
    stanfordnlp.download(demo_language, args.models_dir)
    # set up a pipeline
    print('---')
    print('Building pipeline...')
    pipeline = stanfordnlp.Pipeline(models_dir=args.models_dir, lang=demo_language)
    # process the document
    doc = pipeline('Barack Obama was born in Hawaii.  He was elected president in 2008.')
    # access nlp annotations
    print('')
    print('---')
    print('tokens of first sentence: ')
    for tok in doc.sentences[0].tokens:
        print(tok.word + '\t' + tok.lemma + '\t' + tok.pos)
    print('')
    print('---')
    print('dependency parse of first sentence: ')
    doc.sentences[0].print_dependencies()
    print('')

