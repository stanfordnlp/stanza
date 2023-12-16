import argparse
import os
import re

import stanza
from stanza.models.lemma_classifier import utils

from typing import List, Tuple, Any

"""
The code in this file processes a CoNLL dataset by taking its sentences and filtering out all sentences that do not contain the target token.
Furthermore, it will store tuples of the Stanza document object, the position index of the target token, and its lemma. 
"""


class DataProcessor():

    def __init__(self, target_word: str, target_upos: List[str], allowed_lemmas: str):
        self.target_word = target_word 
        self.target_upos = target_upos
        self.allowed_lemmas = re.compile(allowed_lemmas)
    
    def find_all_occurrences(self, sentence) -> List[int]:
        """
        Finds all occurrences of self.target_word in tokens and returns the index(es) of such occurrences.
        """
        occurrences = []
        for idx, token in enumerate(sentence.words):
            if token.text == self.target_word and token.upos in self.target_upos:
                occurrences.append(idx)
        return occurrences
    
    def process_document(self, doc, keep_condition: callable, save_name: str) -> None:
        """
        Takes any sentence from `doc` that meets the condition of `keep_condition` and writes its tokens, index of target word, and lemma to `save_name`

        Sentences that meet `keep_condition` and contain `self.target_word` multiple times have each instance in a different example in the output file.

        Args:
            doc (Stanza.doc): Document object that represents the file to be analyzed
            keep_condition (callable): A function that outputs a boolean representing whether to analyze (True) or not analyze the sentence for a target word.
            save_name (str): Path to the file for storing output
        """
        with open(save_name, "w+", encoding="utf-8") as output_f:
            for sentence in doc.sentences:
                # for each sentence, we need to determine if it should be added to the output file.
                # if the sentence fulfills the keep_condition, then we will save it along with the target word's index and its corresponding lemma
                if keep_condition(sentence):
                    tokens = [token.text for token in sentence.words]
                    indexes = self.find_all_occurrences(sentence)
                    for idx in indexes:
                        if self.allowed_lemmas.fullmatch(sentence.words[idx].lemma):
                            # for each example found, we write the tokens along with the target word index and lemma
                            output_f.write(f'{" ".join(tokens)} {idx} {sentence.words[idx].lemma}\n')

    def read_processed_data(self, file_name: str) -> List[dict]:
        """
        Reads the output file from `process_document()` and outputs a list that contains the sentences of interest. Each object within the list
        contains a map with three (key, val) pairs:

        "words" is a list that contains the tokens of the sentence
        "index" is an integer representing which token in "words" the lemma annotation corresponds to
        "lemma" is a string that is the lemma of the target word in the sentence.

        """
        output = []
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f.readlines():
                obj = {}
                split = line.split()
                words, index, lemma = split[:-2], int(split[-2]), split[-1]
                
                obj["words"] = words
                obj["index"] = index
                obj["lemma"] = lemma
            
                output.append(obj)

        return output


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--conll_path", type=str, default=os.path.join(os.path.dirname(__file__), "en_gum-ud-train.conllu"), help="path to the conll file to translate")
    parser.add_argument("--target_word", type=str, default="'s", help="Token to classify on, e.g. 's.")
    parser.add_argument("--target_upos", type=str, default="AUX", help="upos on target token")
    parser.add_argument("--output_path", type=str, default="test_output.txt", help="Path for output file")
    parser.add_argument("--allowed_lemmas", type=str, default=".*", help="A regex for allowed lemmas.  If not set, all lemmas are allowed")

    args = parser.parse_args(args)

    conll_path = args.conll_path
    target_word = args.target_word
    target_upos = args.target_upos
    output_path = args.output_path
    allowed_lemmas = args.allowed_lemmas

    doc = utils.load_doc_from_conll_file(conll_path)
    processor = DataProcessor(target_word=target_word, target_upos=[target_upos], allowed_lemmas=allowed_lemmas)
    
    def keep_sentence(sentence):
        for word in sentence.words:
            if word.text == target_word and word.upos == target_upos:
                return True 
        return False

    processor.process_document(doc, keep_sentence, output_path)

    # print(processor.read_processed_data(output_path))  # print out examples

if __name__ == "__main__":
    main()
