"""
Utils for the processing of NER datasets

These can be invoked from either the specific dataset scripts
or the entire prepare_ner_dataset.py script
"""

from collections import defaultdict
import json
import os
import random

from stanza.models.common.doc import Document
import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file

SHARDS = ('train', 'dev', 'test')

def bioes_to_bio(tags):
    new_tags = []
    in_entity = False
    for tag in tags:
        if tag == 'O':
            new_tags.append(tag)
            in_entity = False
        elif in_entity and (tag.startswith("B-") or tag.startswith("S-")):
            # TODO: does the tag have to match the previous tag?
            # eg, does B-LOC B-PER in BIOES need a B-PER or is I-PER sufficient?
            new_tags.append('B-' + tag[2:])
        else:
            new_tags.append('I-' + tag[2:])
            in_entity = True
    return new_tags

def convert_bioes_to_bio(base_input_path, base_output_path, short_name):
    """
    Convert BIOES files back to BIO (not BIO2)

    Useful for preparing datasets for CoreNLP, which doesn't do great with the more highly split classes
    """
    for shard in SHARDS:
        input_filename = os.path.join(base_input_path, '%s.%s.bioes' % (short_name, shard))
        output_filename = os.path.join(base_output_path, '%s.%s.bio' % (short_name, shard))

        input_sentences = read_tsv(input_filename, text_column=0, annotation_column=1)
        new_sentences = []
        for sentence in input_sentences:
            tags = [x[1] for x in sentence]
            tags = bioes_to_bio(tags)
            sentence = [(x[0], y) for x, y in zip(sentence, tags)]
            new_sentences.append(sentence)
        write_sentences(output_filename, new_sentences)


def convert_bio_to_json(base_input_path, base_output_path, short_name, suffix="bio", shard_names=SHARDS, shards=SHARDS):
    """
    Convert BIO files to json

    It can often be convenient to put the intermediate BIO files in
    the same directory as the output files, in which case you can pass
    in same path for both base_input_path and base_output_path.

    This also will rewrite a BIOES as json
    """
    for input_shard, output_shard in zip(shard_names, shards):
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
        for sent_idx, sentence in enumerate(dataset):
            for word_idx, word in enumerate(sentence):
                if len(word) > 2:
                    word = word[:2]
                try:
                    fout.write("%s\t%s\n" % word)
                except TypeError:
                    raise TypeError("Unable to process sentence %d word %d of file %s" % (sent_idx, word_idx, output_filename))
            fout.write("\n")

def write_dataset(datasets, output_dir, short_name, suffix="bio", shard_names=SHARDS, shards=SHARDS):
    """
    write all three pieces of a dataset to output_dir

    datasets should be 3 lists: train, dev, test
    each list should be a list of sentences
    each sentence is a list of pairs: word, tag

    after writing to .bio files, the files will be converted to .json
    """
    for shard, dataset in zip(shard_names, datasets):
        output_filename = os.path.join(output_dir, "%s.%s.%s" % (short_name, shard, suffix))
        write_sentences(output_filename, dataset)

    convert_bio_to_json(output_dir, output_dir, short_name, suffix, shard_names=shard_names, shards=shards)


def write_multitag_json(output_filename, dataset):
    json_dataset = []
    for sentence in dataset:
        json_sentence = []
        for word in sentence:
            word = {'text': word[0],
                    'ner': word[1],
                    'multi_ner': word[2]}
            json_sentence.append(word)
        json_dataset.append(json_sentence)
    with open(output_filename, 'w', encoding='utf-8') as fout:
        json.dump(json_dataset, fout, indent=2)

def write_multitag_dataset(datasets, output_dir, short_name, suffix="bio", shard_names=SHARDS, shards=SHARDS):
    for shard, dataset in zip(shard_names, datasets):
        output_filename = os.path.join(output_dir, "%s.%s.%s" % (short_name, shard, suffix))
        write_sentences(output_filename, dataset)

    for shard, dataset in zip(shard_names, datasets):
        output_filename = os.path.join(output_dir, "%s.%s.json" % (short_name, shard))
        write_multitag_json(output_filename, dataset)

def read_tsv(filename, text_column, annotation_column, remap_fn=None, skip_comments=True, keep_broken_tags=False, keep_all_columns=False, separator="\t"):
    """
    Read sentences from a TSV file

    Returns a list of list of (word, tag)

    If keep_broken_tags==True, then None is returned for a missing.  Otherwise, an IndexError is thrown
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

        pieces = line.split(separator)
        try:
            word = pieces[text_column]
        except IndexError as e:
            raise IndexError("Could not find word index %d at line %d |%s|" % (text_column, line_idx, line)) from e
        if word == '\x96':
            # this happens in GermEval2014 for some reason
            continue
        try:
            tag = pieces[annotation_column]
        except IndexError as e:
            if keep_broken_tags:
                tag = None
            else:
                raise IndexError("Could not find tag index %d at line %d |%s|" % (annotation_column, line_idx, line)) from e
        if remap_fn:
            tag = remap_fn(tag)

        if keep_all_columns:
            pieces[annotation_column] = tag
            current_sentence.append(pieces)
        else:
            current_sentence.append((word, tag))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences

def random_shuffle_directory(input_dir, output_dir, short_name):
    input_files = os.listdir(input_dir)
    input_files = sorted(input_files)
    random_shuffle_files(input_dir, input_files, output_dir, short_name)

def random_shuffle_files(input_dir, input_files, output_dir, short_name):
    """
    Shuffle the files into different chunks based on their filename

    The first piece of the filename, split by ".", is used as a random seed.

    This will make it so that adding new files or using a different
    annotation scheme (assuming that's encoding in pieces of the
    filename) won't change the distibution of the files
    """
    input_keys = {}
    for f in input_files:
        seed = f.split(".")[0]
        if seed in input_keys:
            raise ValueError("Multiple files with the same prefix: %s and %s" % (input_keys[seed], f))
        input_keys[seed] = f
    assert len(input_keys) == len(input_files)

    train_files = []
    dev_files = []
    test_files = []

    for filename in input_files:
        seed = filename.split(".")[0]
        # "salt" the filenames when using as a seed
        # definitely not because of a dumb bug in the original implementation
        seed = seed + ".txt.4class.tsv"
        random.seed(seed, 2)
        location = random.random()
        if location < 0.7:
            train_files.append(filename)
        elif location < 0.8:
            dev_files.append(filename)
        else:
            test_files.append(filename)

    print("Train files: %d  Dev files: %d  Test files: %d" % (len(train_files), len(dev_files), len(test_files)))
    assert len(train_files) + len(dev_files) + len(test_files) == len(input_files)

    file_lists = [train_files, dev_files, test_files]
    datasets = []
    for files in file_lists:
        dataset = []
        for filename in files:
            dataset.extend(read_tsv(os.path.join(input_dir, filename), 0, 1))
        datasets.append(dataset)

    write_dataset(datasets, output_dir, short_name)
    return len(train_files), len(dev_files), len(test_files)

def random_shuffle_by_prefixes(input_dir, output_dir, short_name, prefix_map):
    input_files = os.listdir(input_dir)
    input_files = sorted(input_files)

    file_divisions = defaultdict(list)
    for filename in input_files:
        for division in prefix_map.keys():
            for prefix in prefix_map[division]:
                if filename.startswith(prefix):
                    break
            else: # for/else is intentional
                continue
            break
        else: # yes, stop asking
            raise ValueError("Could not assign %s to any of the divisions in the prefix_map" % filename)
        #print("Assigning %s to %s because of %s" % (filename, division, prefix))
        file_divisions[division].append(filename)

    num_train_files = 0
    num_dev_files = 0
    num_test_files = 0
    for division in file_divisions.keys():
        print()
        print("Processing %d files from %s" % (len(file_divisions[division]), division))
        d_train, d_dev, d_test = random_shuffle_files(input_dir, file_divisions[division], output_dir, "%s-%s" % (short_name, division))
        num_train_files += d_train
        num_dev_files += d_dev
        num_test_files += d_test

    print()
    print("After shuffling: Train files: %d  Dev files: %d  Test files: %d" % (num_train_files, num_dev_files, num_test_files))
    dataset_divisions = ["%s-%s" % (short_name, division) for division in file_divisions]
    combine_dataset(output_dir, output_dir, dataset_divisions, short_name)

def combine_dataset(input_dir, output_dir, input_datasets, output_dataset):
    datasets = []
    for shard in SHARDS:
        full_dataset = []
        for input_dataset in input_datasets:
            input_filename = "%s.%s.json" % (input_dataset, shard)
            input_path = os.path.join(input_dir, input_filename)
            with open(input_path, encoding="utf-8") as fin:
                dataset = json.load(fin)
                converted = [[(word['text'], word['ner']) for word in sentence] for sentence in dataset]
                full_dataset.extend(converted)
        datasets.append(full_dataset)
    write_dataset(datasets, output_dir, output_dataset)

def read_prefix_file(destination_file):
    """
    Read a prefix file such as the one for the Worldwide dataset

    the format should be

    africa:
    af_
    ...

    asia:
    cn_
    ...
    """
    destination = None
    known_prefixes = set()
    prefixes = []

    prefix_map = {}
    with open(destination_file, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                continue
            if line.endswith(":"):
                if destination is not None:
                    prefix_map[destination] = prefixes
                prefixes = []
                destination = line[:-1].strip().lower().replace(" ", "_")
            else:
                if not destination:
                    raise RuntimeError("Found a prefix before the first label was assigned when reading %s" % destination_file)
                prefixes.append(line)
                if line in known_prefixes:
                    raise RuntimeError("Found the same prefix twice! %s" % line)
                known_prefixes.add(line)

        if destination and prefixes:
            prefix_map[destination] = prefixes

    return prefix_map

def read_json_entities(filename):
    """
    Read entities from a file, return a list of (text, label)

    Should work on both BIOES and BIO
    """
    with open(filename) as fin:
        doc = Document(json.load(fin))

        return list_doc_entities(doc)

def list_doc_entities(doc):
    """
    Return a list of (text, label)

    Should work on both BIOES and BIO
    """
    entities = []
    for sentence in doc.sentences:
        current_entity = []
        previous_label = None
        for token in sentence.tokens:
            if token.ner == 'O' or token.ner.startswith("E-"):
                if token.ner.startswith("E-"):
                    current_entity.append(token.text)
                if current_entity:
                    assert previous_label is not None
                    entities.append((current_entity, previous_label))
                    current_entity = []
                    previous_label = None
            elif token.ner.startswith("I-"):
                if previous_label is not None and previous_label != 'O' and previous_label != token.ner[2:]:
                    if current_entity:
                        assert previous_label is not None
                        entities.append((current_entity, previous_label))
                        current_entity = []
                        previous_label = token.ner[2:]
                current_entity.append(token.text)
            elif token.ner.startswith("B-") or token.ner.startswith("S-"):
                if current_entity:
                    assert previous_label is not None
                    entities.append((current_entity, previous_label))
                    current_entity = []
                    previous_label = None
                current_entity.append(token.text)
                previous_label = token.ner[2:]
                if token.ner.startswith("S-"):
                    assert previous_label is not None
                    entities.append(current_entity)
                    current_entity = []
                    previous_label = None
            else:
                raise RuntimeError("Expected BIO(ES) format in the json file!")
            previous_label = token.ner[2:]
        if current_entity:
            assert previous_label is not None
            entities.append((current_entity, previous_label))
    entities = [(tuple(x[0]), x[1]) for x in entities]
    return entities

def combine_files(output_filename, *input_filenames):
    """
    Combine multiple NER json files into one NER file
    """
    doc = []

    for filename in input_filenames:
        with open(filename) as fin:
            new_doc = json.load(fin)
            doc.extend(new_doc)

    with open(output_filename, "w") as fout:
        json.dump(doc, fout, indent=2)

