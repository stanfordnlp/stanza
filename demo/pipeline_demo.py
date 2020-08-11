"""
A basic demo of the Stanza neural pipeline.
"""

import sys
import argparse
import os

import stanza
from stanza.resources.common import DEFAULT_MODEL_DIR


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--models_dir', help='location of models files | default: ~/stanza_resources',
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument('-l', '--lang', help='Demo language',
                        default="en")
    parser.add_argument('-c', '--cpu', action='store_true', help='Use cpu as the device.')
    args = parser.parse_args()

    example_sentences = {"en": "Barack Obama was born in Hawaii.  He was elected president in 2008.",
            "zh": "中国文化经历上千年的历史演变，是各区域、各民族古代文化长期相互交流、借鉴、融合的结果。",
            "fr": "Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie. Il tente d'abord de faire carrière comme marchand d'art chez Goupil & C.",
            "vi": "Trận Trân Châu Cảng (hay Chiến dịch Hawaii theo cách gọi của Bộ Tổng tư lệnh Đế quốc Nhật Bản) là một đòn tấn công quân sự bất ngờ được Hải quân Nhật Bản thực hiện nhằm vào căn cứ hải quân của Hoa Kỳ tại Trân Châu Cảng thuộc tiểu bang Hawaii vào sáng Chủ Nhật, ngày 7 tháng 12 năm 1941, dẫn đến việc Hoa Kỳ sau đó quyết định tham gia vào hoạt động quân sự trong Chiến tranh thế giới thứ hai."}

    if args.lang not in example_sentences:
        print(f'Sorry, but we don\'t have a demo sentence for "{args.lang}" for the moment. Try one of these languages: {list(example_sentences.keys())}')
        sys.exit(1)

    # download the models
    stanza.download(args.lang, dir=args.models_dir)
    # set up a pipeline
    print('---')
    print('Building pipeline...')
    pipeline = stanza.Pipeline(lang=args.lang, dir=args.models_dir, use_gpu=(not args.cpu))
    # process the document
    doc = pipeline(example_sentences[args.lang])
    # access nlp annotations
    print('')
    print('Input: {}'.format(example_sentences[args.lang]))
    print("The tokenizer split the input into {} sentences.".format(len(doc.sentences)))
    print('---')
    print('tokens of first sentence: ')
    doc.sentences[0].print_tokens()
    print('')
    print('---')
    print('dependency parse of first sentence: ')
    doc.sentences[0].print_dependencies()
    print('')

