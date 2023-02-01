"""
Script which processes a dependency file by using the constituency parser, then converting with the CoreNLP converter

Currently this does not have the constituency parser as an option,
although that is easy to add.

Only English is supported, as only English is available in the CoreNLP converter
"""

import argparse
import os
import tempfile

import stanza
from stanza.models.constituency import retagging
from stanza.models.depparse import scorer
from stanza.utils.conll import CoNLL

def score_dependencies(args):
    if args['lang'] != 'en':
        raise ValueError("Converting and scoring dependencies is currently only supported for English")

    constituency_package = 'wsj_bert'
    pipeline_args = {'lang': args['lang'],
                     'tokenize_pretokenized': True,
                     'package': {'pos': args['retag_package'], 'depparse': 'converter', 'constituency': constituency_package},
                     'processors': 'tokenize, pos, constituency, depparse'}
    pipeline = stanza.Pipeline(**pipeline_args)

    input_doc = CoNLL.conll2doc(args['eval_file'])
    output_doc = pipeline(input_doc)
    print("Processed %d sentences" % len(output_doc.sentences))
    # reload - the pipeline clobbered the gold values
    input_doc = CoNLL.conll2doc(args['eval_file'])

    scorer.score_named_dependencies(output_doc, input_doc)
    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, "converted.conll")

        CoNLL.write_doc2conll(output_doc, output_path)

        _, _, score = scorer.score(output_path, args['eval_file'])

        print("Parser score:")
        print("{} {:.2f}".format(constituency_package, score*100))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lang', default='en', type=str, help='Language')
    parser.add_argument('--eval_file', default="extern_data/ud2/ud-treebanks-v2.11/UD_English-EWT/en_ewt-ud-test.conllu", help='Input file for data loader.')

    retagging.add_retag_args(parser)
    args = parser.parse_args()

    args = vars(args)
    retagging.postprocess_args(args)

    score_dependencies(args)

if __name__ == '__main__':
    main()
    
