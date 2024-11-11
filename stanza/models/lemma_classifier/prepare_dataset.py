import argparse
import json
import os
import re

import stanza
from stanza.models.lemma_classifier import utils

from typing import List, Tuple, Any

"""
The code in this file processes a CoNLL dataset by taking its sentences and filtering out all sentences that do not contain the target token.
Furthermore, it will store tuples of the Stanza document object, the position index of the target token, and its lemma.
"""


def load_doc_from_conll_file(path: str):
    """"
    loads in a Stanza document object from a path to a CoNLL file containing annotated sentences.
    """
    return stanza.utils.conll.CoNLL.conll2doc(path)


class DataProcessor():

    def __init__(self, target_word: str, target_upos: List[str], allowed_lemmas: str):
        self.target_word = target_word
        self.target_word_regex = re.compile(target_word)
        self.target_upos = target_upos
        self.allowed_lemmas = re.compile(allowed_lemmas)

    def keep_sentence(self, sentence):
        for word in sentence.words:
            if self.target_word_regex.fullmatch(word.text) and word.upos in self.target_upos:
                return True
        return False

    def find_all_occurrences(self, sentence) -> List[int]:
        """
        Finds all occurrences of self.target_word in tokens and returns the index(es) of such occurrences.
        """
        occurrences = []
        for idx, token in enumerate(sentence.words):
            if self.target_word_regex.fullmatch(token.text) and token.upos in self.target_upos:
                occurrences.append(idx)
        return occurrences

    @staticmethod
    def write_output_file(save_name, target_upos, sentences):
        with open(save_name, "w+", encoding="utf-8") as output_f:
            output_f.write("{\n")
            output_f.write('  "upos": %s,\n' % json.dumps(target_upos))
            output_f.write('  "sentences": [')
            wrote_sentence = False
            for sentence in sentences:
                if not wrote_sentence:
                    output_f.write("\n    ")
                    wrote_sentence = True
                else:
                    output_f.write(",\n    ")
                output_f.write(json.dumps(sentence))
            output_f.write("\n  ]\n}\n")

    def process_document(self, doc, save_name: str) -> None:
        """
        Takes any sentence from `doc` that meets the condition of `keep_sentence` and writes its tokens, index of target word, and lemma to `save_name`

        Sentences that meet `keep_sentence` and contain `self.target_word` multiple times have each instance in a different example in the output file.

        Args:
            doc (Stanza.doc): Document object that represents the file to be analyzed
            save_name (str): Path to the file for storing output
        """
        sentences = []
        for sentence in doc.sentences:
            # for each sentence, we need to determine if it should be added to the output file.
            # if the sentence fulfills keep_sentence, then we will save it along with the target word's index and its corresponding lemma
            if self.keep_sentence(sentence):
                tokens = [token.text for token in sentence.words]
                indexes = self.find_all_occurrences(sentence)
                for idx in indexes:
                    if self.allowed_lemmas.fullmatch(sentence.words[idx].lemma):
                        # for each example found, we write the tokens,
                        # their respective upos tags, the target token index,
                        # and the target lemma
                        upos_tags = [sentence.words[i].upos for i in range(len(sentence.words))]
                        num_tokens = len(upos_tags)
                        sentences.append({
                            "words": tokens,
                            "upos_tags": upos_tags,
                            "index": idx,
                            "lemma": sentence.words[idx].lemma
                        })

        if save_name:
            self.write_output_file(save_name, self.target_upos, sentences)
        return sentences

def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--conll_path", type=str, default=os.path.join(os.path.dirname(__file__), "en_gum-ud-train.conllu"), help="path to the conll file to translate")
    parser.add_argument("--target_word", type=str, default="'s", help="Token to classify on, e.g. 's.")
    parser.add_argument("--target_upos", type=str, default="AUX", help="upos on target token")
    parser.add_argument("--output_path", type=str, default="test_output.txt", help="Path for output file")
    parser.add_argument("--allowed_lemmas", type=str, default=".*", help="A regex for allowed lemmas.  If not set, all lemmas are allowed")

    args = parser.parse_args(args)

    conll_path = args.conll_path
    target_upos = args.target_upos
    output_path = args.output_path
    allowed_lemmas = args.allowed_lemmas

    args = vars(args)
    for arg in args:
        print(f"{arg}: {args[arg]}")

    doc = load_doc_from_conll_file(conll_path)
    processor = DataProcessor(target_word=args['target_word'], target_upos=[target_upos], allowed_lemmas=allowed_lemmas)

    return processor.process_document(doc, output_path)

if __name__ == "__main__":
    main()
