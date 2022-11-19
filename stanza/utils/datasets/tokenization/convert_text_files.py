"""
Given a text file and a file with one word per line, convert the text file

Sentence splits should be represented as blank lines at the end of a sentence.
"""

import argparse
import os
import random

from stanza.models.tokenization.utils import match_tokens_with_text
import stanza.utils.datasets.common as common

def read_tokens_file(token_file):
    """
    Returns a list of list of tokens

    Each sentence is a list of tokens
    """
    sentences = []
    current_sentence = []
    with open(token_file, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
        if current_sentence:
            sentences.append(current_sentence)

    return sentences

def read_sentences_file(sentence_file):
    sentences = []
    with open(sentence_file, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sentences.append(line)
    return sentences

def process_raw_file(text_file, token_file, sentence_file):
    """
    Process a text file separated into a list of tokens using match_tokens_with_text from the tokenizer

    The tokens are one per line in the token_file
    The tokens in the token_file must add up to the text_file modulo whitespace.

    Sentences are also one per line in the sentence_file
    These must also add up to text_file

    The return format is a list of list of conllu lines representing the sentences.
    The only fields set will be the token index, the token text, and possibly SpaceAfter=No
    where SpaceAfter=No is true if the next token started with no whitespace in the text file
    """
    with open(text_file, encoding="utf-8") as fin:
        text = fin.read()

    tokens = read_tokens_file(token_file)
    tokens = [[token for sentence in tokens for token in sentence]]
    tokens_doc = match_tokens_with_text(tokens, text)

    assert len(tokens_doc.sentences) == 1
    assert len(tokens_doc.sentences[0].tokens) == len(tokens[0])

    sentences = read_sentences_file(sentence_file)
    sentences_doc = match_tokens_with_text([sentences], text)

    assert len(sentences_doc.sentences) == 1
    assert len(sentences_doc.sentences[0].tokens) == len(sentences)

    start_token_idx = 0
    sentences = []
    for sent_idx, sentence in enumerate(sentences_doc.sentences[0].tokens):
        tokens = []
        tokens.append("# sent_id = %d" % (sent_idx+1))
        tokens.append("# text = %s" % text[sentence.start_char:sentence.end_char].replace("\n", " "))
        token_idx = 0
        while token_idx + start_token_idx < len(tokens_doc.sentences[0].tokens):
            token = tokens_doc.sentences[0].tokens[token_idx + start_token_idx]
            if token.start_char >= sentence.end_char:
                # have reached the end of this sentence
                # continue with the next sentence
                start_token_idx += token_idx
                break

            if token_idx + start_token_idx == len(tokens_doc.sentences[0].tokens) - 1:
                # definitely the end of the document
                space_after = True
            elif token.end_char == tokens_doc.sentences[0].tokens[token_idx + start_token_idx + 1].start_char:
                space_after = False
            else:
                space_after = True
            token = [str(token_idx+1), token.text] + ["_"] * 7 + ["_" if space_after else "SpaceAfter=No"]
            assert len(token) == 10, "Token length: %d" % len(token)
            token = "\t".join(token)
            tokens.append(token)
            token_idx += 1
        sentences.append(tokens)
    return sentences

def extract_sentences(dataset_files):
    sentences = []
    for text_file, token_file, sentence_file in dataset_files:
        print("Extracting sentences from %s and tokens from %s from the text file %s" % (sentence_file, token_file, text_file))
        sentences.extend(process_raw_file(text_file, token_file, sentence_file))
    return sentences

def split_sentences(sentences, train_split=0.8, dev_split=0.1):
    """
    Splits randomly without shuffling
    """
    generator = random.Random(1234)

    train = []
    dev = []
    test = []
    for sentence in sentences:
        r = generator.random()
        if r < train_split:
            train.append(sentence)
        elif r < train_split + dev_split:
            dev.append(sentence)
        else:
            test.append(sentence)
    return (train, dev, test)

def find_dataset_files(input_path, token_prefix, sentence_prefix):
    files = os.listdir(input_path)
    print("Found %d files in %s" % (len(files), input_path))
    if len(files) > 0:
        if len(files) < 20:
            print("Files:", end="\n  ")
        else:
            print("First few files:", end="\n  ")
        print("\n  ".join(files[:20]))
    token_files = {}
    sentence_files = {}
    text_files = []
    for filename in files:
        if filename.endswith(".zip"):
            continue
        if filename.startswith(token_prefix):
            short_filename = filename[len(token_prefix):]
            if short_filename.startswith("_"):
                short_filename = short_filename[1:]
            token_files[short_filename] = filename
        elif filename.startswith(sentence_prefix):
            short_filename = filename[len(sentence_prefix):]
            if short_filename.startswith("_"):
                short_filename = short_filename[1:]
            sentence_files[short_filename] = filename
        else:
            text_files.append(filename)
    dataset_files = []
    for filename in text_files:
        if filename not in token_files:
            raise FileNotFoundError("When looking in %s, found %s as a text file, but did not find a corresponding tokens file at %s_%s  Please give an input directory which has only the text files, tokens files, and sentences files" % (input_path, filename, token_prefix, filename))
        if filename not in sentence_files:
            raise FileNotFoundError("When looking in %s, found %s as a text file, but did not find a corresponding sentences file at %s_%s  Please give an input directory which has only the text files, tokens files, and sentences files" % (input_path, filename, sentence_prefix, filename))
        text_file = os.path.join(input_path, filename)
        token_file = os.path.join(input_path, token_files[filename])
        sentence_file = os.path.join(input_path, sentence_files[filename])
        dataset_files.append((text_file, token_file, sentence_file))
    return dataset_files

SHARDS = ("train", "dev", "test")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_prefix', type=str, default="tkns", help="Prefix for the token files")
    parser.add_argument('--sentence_prefix', type=str, default="stns", help="Prefix for the token files")
    parser.add_argument('--input_path', type=str, default="extern_data/sindhi/tokenization", help="Where to find all of the input files.  Files with the prefix tkns_ will be treated as token files, files with the prefix stns_ will be treated as sentence files, and all others will be the text files.")
    parser.add_argument('--output_path', type=str, default="data/tokenize", help="Where to output the results")
    parser.add_argument('--dataset', type=str, default="sd_isra", help="What name to give this dataset")
    args = parser.parse_args()

    dataset_files = find_dataset_files(args.input_path, args.token_prefix, args.sentence_prefix)

    tokenizer_dir = args.output_path
    short_name = args.dataset  # todo: convert a full name?

    sentences = extract_sentences(dataset_files)
    splits = split_sentences(sentences)

    os.makedirs(args.output_path, exist_ok=True)
    for dataset, shard in zip(splits, SHARDS):
        output_conllu = common.tokenizer_conllu_name(tokenizer_dir, short_name, shard)
        common.write_sentences_to_conllu(output_conllu, dataset)

    common.convert_conllu_to_txt(tokenizer_dir, short_name)
    common.prepare_tokenizer_treebank_labels(tokenizer_dir, short_name)

if __name__ == '__main__':
    main()
