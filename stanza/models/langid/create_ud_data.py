"""
Script for producing training/dev/test data from UD data
"""

import argparse
import json
import os
import re
import sys

from pathlib import Path
from random import randint, random, shuffle
from string import digits
from tqdm import tqdm

DEFAULT_LANGUAGES = ["da", "en", "sv", "no", "de", "cs", "es", "fr", "pt", "it", "tr", "nl", "fi", "pl", "ro", "hu", "lt",
                     "ca", "hr", "et"]


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-length", help="length of eval strings", type=int, default=10)
    parser.add_argument("--languages", help="list of languages to use, or \"all\"", default=DEFAULT_LANGUAGES)
    parser.add_argument("--ud-path", help="path to ud data")
    parser.add_argument("--save-path", help="path to save data", default=".")
    args = parser.parse_args(args=args)
    return args


def main(args=None):
    args = parse_args(args=args)
    if isinstance(args.languages, str):
        args.languages = args.languages.split(",")
    data_paths = [f"{args.save_path}/{data_split}.jsonl" for data_split in ["train", "dev", "test"]]
    lang_to_files = collect_files(args.ud_path, args.languages)
    print(f"Building UD data for languages: {','.join(args.languages)}")
    for lang_id in tqdm(lang_to_files):
        lang_examples = generate_examples(lang_id, lang_to_files[lang_id], eval_length=args.eval_length)
        for (data_set, save_path) in zip(lang_examples, data_paths):
            with open(save_path, "a") as json_file:
                for json_entry in data_set:
                    json.dump(json_entry, json_file, ensure_ascii=False)
                    json_file.write("\n")


def collect_files(ud_path, languages):
    """ 
    Given path to UD, collect files 
    """
    ud_files = Path(ud_path).glob("*/*.conllu")
    lang_to_files = {}
    for ud_file in ud_files:
        lang_id = ud_file.name[:2]
        if lang_id not in languages and "all" not in languages:
            continue
        if not lang_id in lang_to_files:
            lang_to_files[lang_id] = []
        lang_to_files[lang_id].append(ud_file)
    return lang_to_files


def generate_examples(lang_id, list_of_files, splits=(0.8,0.1,0.1), min_window=10, max_window=50, 
                      eval_length=10):
    """
    Generate train/dev/test examples for a given language
    """
    examples = []
    for ud_file in list_of_files:
        sentences = sentences_from_file(ud_file)
        for sentence in sentences:
            sentence = clean_sentence(sentence)
            if validate_sentence(sentence, min_window):
                examples += sentence_to_windows(sentence, min_window=min_window, max_window=max_window)
    shuffle(examples)
    train_idx = int(splits[0] * len(examples))
    train_set = [example_json(lang_id, example) for example in examples[:train_idx]]
    dev_idx = int(splits[1] * len(examples)) + train_idx
    dev_set = [example_json(lang_id, example, eval_length=eval_length) for example in examples[train_idx:dev_idx]]
    test_set = [example_json(lang_id, example, eval_length=eval_length) for example in examples[dev_idx:]]
    return train_set, dev_set, test_set


def sentences_from_file(ud_file_path):
    """
    Retrieve all sentences from a UD file
    """
    all_sentences = []
    with open(ud_file_path) as ud_file:
        sentences = [x[9:] for x in ud_file.read().split("\n") if x.startswith("# text = ")]
        all_sentences += sentences
    return all_sentences


def sentence_to_windows(sentence, min_window, max_window):
    """
    Create window size chunks from a sentence, always starting with a word
    """
    windows = []
    words = sentence.split(" ")
    curr_window = ""
    for idx, word in enumerate(words):
        curr_window += (" " + word)
        curr_window = curr_window.lstrip()
        next_word_len = len(words[idx+1]) + 1 if idx+1 < len(words) else 0
        if len(curr_window) + next_word_len > max_window:
            curr_window = clean_sentence(curr_window)
            if validate_sentence(curr_window, min_window):
                windows.append(curr_window.strip())
            curr_window = ""
    if len(curr_window) >= min_window:
        windows.append(curr_window)
    return windows


def validate_sentence(current_window, min_window):
    """
    Sentence validation from: LSTM-LID
    GitHub: https://github.com/AU-DIS/LSTM_langid/blob/main/src/dataset_creator.py
    """
    if len(current_window) < min_window:
        return False
    if not re.search('[a-zA-Z]', current_window):
        return False
    return True

def find(s, ch):
    """ 
    Helper for clean_sentence from LSTM-LID
    GitHub: https://github.com/AU-DIS/LSTM_langid/blob/main/src/dataset_creator.py 
    """
    return [i for i, ltr in enumerate(s) if ltr == ch]


def clean_sentence(line):
    """ 
    Sentence cleaning from LSTM-LID
    GitHub: https://github.com/AU-DIS/LSTM_langid/blob/main/src/dataset_creator.py
    """
    # We remove some special characters and fix small errors in the data, to improve the quality of the data
    line = line.replace("\n", '') #{"text": "- Mor.\n", "label": "da"}
    line = line.replace("- ", '') #{"text": "- Mor.", "label": "da"}
    line = line.replace("_", '') #{"text": "- Mor.", "label": "da"}
    line = line.replace("\\", '')
    line = line.replace("\"", '')
    line = line.replace("  ", " ")
    remove_digits = str.maketrans('', '', digits)
    line = line.translate(remove_digits)
    words = line.split()
    new_words = []
    # Below fixes large I instead of l. Does not catch everything, but should also not really make any mistakes either
    for word in words:
        clean_word = word
        s = clean_word
        if clean_word[1:].__contains__("I"):
            indices = find(clean_word, "I")
            for indx in indices:
                if clean_word[indx-1].islower():
                    if len(clean_word) > indx + 1:
                        if clean_word[indx+1].islower():
                            s = s[:indx] + "l" + s[indx + 1:]
                    else:
                        s = s[:indx] + "l" + s[indx + 1:]
        new_words.append(s)
    new_line = " ".join(new_words)
    return new_line


def example_json(lang_id, text, eval_length=None):
    if eval_length is not None:
        text = text[:eval_length]
    return {"text": text.strip(), "label": lang_id}


if __name__ == "__main__":
    main()

