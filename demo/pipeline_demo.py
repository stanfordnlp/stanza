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
    parser.add_argument('-l', '--lang', help='Demo language',
                        default="en")
    args = parser.parse_args()

    example_sentences = {"en": "Barack Obama was born in Hawaii.  He was elected president in 2008.",
            "zh": "達沃斯世界經濟論壇是每年全球政商界領袖聚在一起的年度盛事。",
            "fr": "Vainqueur de Raonic à l'Open d'Australie, le Français Lucas Pouille atteint pour la première fois de sa carrière une demi-finale en Grand Chelem."}

    if args.lang not in example_sentences:
        print(f'Sorry, but we don\'t have a demo sentence for "{args.lang}" for the moment. Try one of these languages: {list(example_sentences.keys())}')
        exit()

    # download the models
    stanfordnlp.download(args.lang, args.models_dir, confirm_if_exists=True)
    # set up a pipeline
    print('---')
    print('Building pipeline...')
    pipeline = stanfordnlp.Pipeline(models_dir=args.models_dir, lang=args.lang)
    # process the document
    doc = pipeline(example_sentences[args.lang])
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

