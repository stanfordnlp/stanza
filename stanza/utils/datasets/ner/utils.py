"""
Utils for the processing of NER datasets

These can be invoked from either the specific dataset scripts
or the entire prepare_ner_dataset.py script
"""

import os

import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file

SHARDS = ('train', 'dev', 'test')

def convert_bio_to_json(base_input_path, base_output_path, short_name, suffix="bio", shard_names=SHARDS):
    """
    Convert BIO files to json

    It can often be convenient to put the intermediate BIO files in
    the same directory as the output files, in which case you can pass
    in same path for both base_input_path and base_output_path.
    """
    for input_shard, output_shard in zip(shard_names, SHARDS):
        input_filename = os.path.join(base_input_path, '%s.%s.%s' % (short_name, input_shard, suffix))
        if not os.path.exists(input_filename):
            alt_filename = os.path.join(base_input_path, '%s.%s' % (input_shard, suffix))
            if os.path.exists(alt_filename):
                input_filename = alt_filename
            else:
                raise FileNotFoundError('Cannot find %s component of %s in %s or %s' % (output_shard, short_name, input_filename, alt_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, output_shard))
        print("Converting %s to %s" % (input_filename, output_filename))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def get_tags(datasets):
    """
    return the set of tags used in these datasets

    datasets is expected to be train, dev, test but could be any list
    """
    tags = set()
    for dataset in datasets:
        for sentence in dataset:
            for word, tag in sentence:
                tags.add(tag)
    return tags

def write_sentences(output_filename, dataset):
    """
    Write exactly one output file worth of dataset
    """
    os.makedirs(os.path.split(output_filename)[0], exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as fout:
        for sentence in dataset:
            for word in sentence:
                fout.write("%s\t%s\n" % word)
            fout.write("\n")

def write_dataset(datasets, output_dir, short_name, suffix="bio"):
    """
    write all three pieces of a dataset to output_dir

    datasets should be 3 lists: train, dev, test
    each list should be a list of sentences
    each sentence is a list of pairs: word, tag

    after writing to .bio files, the files will be converted to .json
    """
    for shard, dataset in zip(SHARDS, datasets):
        output_filename = os.path.join(output_dir, "%s.%s.%s" % (short_name, shard, suffix))
        write_sentences(output_filename, dataset)

    convert_bio_to_json(output_dir, output_dir, short_name, suffix)


def read_tsv(filename, text_column, annotation_column, remap_fn=None, skip_comments=True, keep_broken_tags=False):
    """
    Read sentences from a TSV file

    Returns a list of list of (word, tag)

    If keep_broken_tags==True, then None is returned for that tag.  Otherwise, an IndexError is thrown
    """
    with open(filename, encoding="utf-8") as fin:
        lines = fin.readlines()

    lines = [x.strip() for x in lines]

    sentences = []
    current_sentence = []
    for line_idx, line in enumerate(lines):
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue
        if skip_comments and line.startswith("#"):
            continue

        pieces = line.split("\t")
        try:
            word = pieces[text_column]
        except IndexError as e:
            raise IndexError("Could not find word index %d at line %d" % (text_column, line_idx)) from e
        if word == '\x96':
            # this happens in GermEval2014 for some reason
            continue
        try:
            tag = pieces[annotation_column]
        except IndexError as e:
            if keep_broken_tags:
                tag = None
            else:
                raise IndexError("Could not find tag index %d at line %d" % (annotation_column, line_idx)) from e
        if remap_fn:
            tag = remap_fn(tag)

        current_sentence.append((word, tag))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences

