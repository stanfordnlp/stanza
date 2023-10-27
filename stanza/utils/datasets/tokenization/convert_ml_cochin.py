"""
Convert a Malayalam NER dataset to a tokenization dataset using
the additional labeling provided by TTec's Indian partners

This is still WIP - ongoing discussion with TTec and the team at UFAL
doing the UD Malayalam dataset - but if someone wants the data to
recreate it, feel free to contact Prof. Manning or John Bauer

Data was annotated through Datasaur by TTec - possibly another team
involved, will double check with the annotators.

#1 current issue with the data is a difference in annotation style
observed by the UFAL group.  I believe TTec is working on reannotating
this.

Discussing the first sentence in the first split file:

> I am not sure about the guidelines that the annotators followed, but
> I would not have split നാമജപത്തോടുകൂടി as നാമ --- ജപത്തോടുകൂടി. Because
> they are not multiple syntactic words. I would have done it like
> നാമജപത്തോടു --- കൂടി as കൂടി ('with') can be tagged as ADP. I agree with
> the second MWT വ്യത്യസ്തം --- കൂടാതെ.
>
> In Malayalam, we do have many words which potentially can be treated
> as compounds and split but sometimes it becomes difficult to make
> that decision as the etymology or the word formation process is
> unclear. So for the Malayalam UD annotations I stayed away from
> doing it because I didn't find it necessary and moreover the
> guidelines say that the words should be split into syntactic words
> and not into morphemes.

As for using this script, create a directory extern_data/malayalam/cochin_ner/
The original NER dataset from Cochin University going there:
extern_data/malayalam/cochin_ner/final_ner.txt
The relabeled data from TTEC goes in
extern_data/malayalam/cochin_ner/relabeled_tsv/malayalam_File_1.txt.tsv etc etc

This can be invoked from the command line, or it can be used as part of
stanza/utils/datasets/prepare_tokenizer_treebank.py ml_cochin
in which case the conll splits will be turned into tokenizer labels as well
"""

from difflib import SequenceMatcher
import os
import random
import sys

import stanza.utils.default_paths as default_paths

def read_words(filename):
    with open(filename, encoding="utf-8") as fin:
        text = fin.readlines()
        text = [x.strip().split()[0] if x.strip() else "" for x in text]
        return text

def read_original_text(input_dir):
    original_file = os.path.join(input_dir, "final_ner.txt")
    return read_words(original_file)

def list_relabeled_files(relabeled_dir):
    tsv_files = os.listdir(relabeled_dir)
    assert all(x.startswith("malayalam_File_") and x.endswith(".txt.tsv") for x in tsv_files)
    tsv_files = sorted(tsv_files, key = lambda filename: int(filename.split(".")[0].split("_")[2]))
    return tsv_files

def find_word(original_text, target, start_index, end_index):
    for word in original_text[start_index:end_index]:
        if word == target:
            return True
    return False

def scan_file(original_text, current_index, tsv_file):
    relabeled_text = read_words(tsv_file)
    # for now, at least, we ignore these markers
    relabeled_indices = [idx for idx, x in enumerate(relabeled_text) if x != '$' and x != '^']
    relabeled_text = [x for x in relabeled_text if x != '$' and x != '^']
    diffs = SequenceMatcher(None, original_text, relabeled_text, False)

    blocks = diffs.get_matching_blocks()
    assert blocks[-1].size == 0
    if len(blocks) == 1:
        raise ValueError("Could not find a match between %s and the original text" % tsv_file)

    sentences = []
    current_sentence = []

    in_mwt = False
    bad_sentence = False
    current_mwt = []
    block_index = 0
    current_block = blocks[0]
    for tsv_index, next_word in enumerate(relabeled_text):
        if not next_word:
            if in_mwt:
                current_mwt = []
                in_mwt = False
                bad_sentence = True
                print("Unclosed MWT found at %s line %d" % (tsv_file, tsv_index))
            if current_sentence:
                if not bad_sentence:
                    sentences.append(current_sentence)
                bad_sentence = False
                current_sentence = []
            continue

        # tsv_index will now be inside the current block or before the current block
        while tsv_index >= blocks[block_index].b + current_block.size:
            block_index += 1
            current_block = blocks[block_index]
        #print(tsv_index, current_block.b, current_block.size)

        if next_word == ',' or next_word == '.':
            # many of these punctuations were added by the relabelers
            current_sentence.append(next_word)
            continue
        if tsv_index >= current_block.b and tsv_index < current_block.b + current_block.size:
            # ideal case: in a matching block
            current_sentence.append(next_word)
            continue

        # in between blocks... need to handle re-spelled words and MWTs
        if not in_mwt and next_word == '@':
            in_mwt = True
            continue
        if not in_mwt:
            current_sentence.append(next_word)
            continue
        if in_mwt and next_word == '@' and (tsv_index + 1 < len(relabeled_text) and relabeled_text[tsv_index+1] == '@'):
            # we'll stop the MWT next time around
            continue
        if in_mwt and next_word == '@':
            if block_index > 0 and (len(current_mwt) == 2 or len(current_mwt) == 3):
                mwt = "".join(current_mwt)
                start_original = blocks[block_index-1].a + blocks[block_index-1].size
                end_original = current_block.a
                if find_word(original_text, mwt, start_original, end_original):
                    current_sentence.append((mwt, current_mwt))
                else:
                    print("%d word MWT %s at %s %d.  Should be somewhere in %d %d" % (len(current_mwt), mwt, tsv_file, relabeled_indices[tsv_index], start_original, end_original))
                    bad_sentence = True
            elif len(current_mwt) > 6:
                raise ValueError("Unreasonably long MWT span in %s at line %d" % (tsv_file, relabeled_indices[tsv_index]))
            elif len(current_mwt) > 3:
                print("%d word sequence, stop being lazy - %s %d" % (len(current_mwt), tsv_file, relabeled_indices[tsv_index]))
                bad_sentence = True
            else:
                # short MWT, but it was at the start of a file, and we don't want to search the whole file for the item
                # TODO, could maybe search the 10 words or so before the start of the block?
                bad_sentence = True
            current_mwt = []
            in_mwt = False
            continue
        # now we know we are in an MWT... TODO
        current_mwt.append(next_word)

    if len(current_sentence) > 0 and not bad_sentence:
        sentences.append(current_sentence)

    return current_index, sentences

def split_sentences(sentences):
    train = []
    dev = []
    test = []

    for sentence in sentences:
        rand = random.random()
        if rand < 0.8:
            train.append(sentence)
        elif rand < 0.9:
            dev.append(sentence)
        else:
            test.append(sentence)

    return train, dev, test

def main(input_dir, tokenizer_dir, relabeled_dir="relabeled_tsv", split_data=True):
    random.seed(1006)

    input_dir = os.path.join(input_dir, "malayalam", "cochin_ner")
    relabeled_dir = os.path.join(input_dir, relabeled_dir)
    tsv_files = list_relabeled_files(relabeled_dir)

    original_text = read_original_text(input_dir)
    print("Original text len: %d" %len(original_text))
    current_index = 0
    sentences = []
    for tsv_file in tsv_files:
        print(tsv_file)
        current_index, new_sentences = scan_file(original_text, current_index, os.path.join(relabeled_dir, tsv_file))
        sentences.extend(new_sentences)

    print("Found %d sentences" % len(sentences))

    if split_data:
        splits = split_sentences(sentences)
        SHARDS = ("train", "dev", "test")
    else:
        splits = [sentences]
        SHARDS = ["train"]

    for split, shard in zip(splits, SHARDS):
        output_filename = os.path.join(tokenizer_dir, "ml_cochin.%s.gold.conllu" % shard)
        print("Writing %d sentences to %s" % (len(split), output_filename))
        with open(output_filename, "w", encoding="utf-8") as fout:
            for sentence in split:
                word_idx = 1
                for token in sentence:
                    if isinstance(token, str):
                        fake_dep = "\t0\troot" if word_idx == 1 else "\t1\tdep"
                        fout.write("%d\t%s" % (word_idx, token) + "\t_" * 4 + fake_dep + "\t_\t_\n")
                        word_idx += 1
                    else:
                        text = token[0]
                        mwt = token[1]
                        fout.write("%d-%d\t%s" % (word_idx, word_idx + len(mwt) - 1, text) + "\t_" * 8 + "\n")
                        for piece in mwt:
                            fake_dep = "\t0\troot" if word_idx == 1 else "\t1\tdep"
                            fout.write("%d\t%s" % (word_idx, piece) + "\t_" * 4 + fake_dep + "\t_\t_\n")
                            word_idx += 1
                fout.write("\n")

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    paths = default_paths.get_default_paths()
    tokenizer_dir = paths["TOKENIZE_DATA_DIR"]
    input_dir = paths["STANZA_EXTERN_DIR"]
    main(input_dir, tokenizer_dir, "relabeled_tsv_v2", False)

