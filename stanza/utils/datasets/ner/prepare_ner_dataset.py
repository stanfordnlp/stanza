"""
Converts raw data files into json files usable by the training script.

Currently it supports converting wikiner datasets, available here:

https://figshare.com/articles/dataset/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500

Also, Finnish Turku dataset, available here:

https://turkunlp.org/fin-ner.html

IJCNLP 2008 produced a few Indian language NER datasets.
  description:
    http://ltrc.iiit.ac.in/ner-ssea-08/index.cgi?topic=3
  download:
    http://ltrc.iiit.ac.in/ner-ssea-08/index.cgi?topic=5
  The models produced from these datasets have extremely low recall, unfortunately.

FIRE 2013 also produced NER datasets for Indian languages.
  http://au-kbc.org/nlp/NER-FIRE2013/index.html
  The datasets are password locked.
  For Stanford users, contact Chris Manning for license details.
  For external users, please contact the organizers for more information.

Ukranian NER is provided by lang-uk, available here:
  https://github.com/lang-uk/ner-uk
  git clone the repo to $NERBASE/lang-uk
  There should be a subdirectory $NERBASE/lang-uk/ner-uk/data at that point
  Conversion script graciously provided by Andrii Garkavyi @gawy

There are two Hungarian datasets are available here:
  https://rgai.inf.u-szeged.hu/node/130
  http://www.lrec-conf.org/proceedings/lrec2006/pdf/365_pdf.pdf
  We combined them and give them the label hu_rgai
  You can also build individual pieces with hu_rgai_business or hu_rgai_criminal
  Create a subdirectory of $NERBASE, $NERBASE/hu_rgai, and download both of
    the pieces and unzip them in that directory.

Another Hungarian dataset is here:
  https://github.com/nytud/NYTK-NerKor
  git clone the entire thing in your $NERBASE directory to operate on it
"""

import glob
import os
import random
import sys

from stanza.models.common.constant import treebank_to_short_name, lcode2lang
import stanza.utils.default_paths as default_paths

from stanza.utils.datasets.ner.convert_fire_2013 import convert_fire_2013
from stanza.utils.datasets.ner.preprocess_wikiner import preprocess_wikiner
from stanza.utils.datasets.ner.split_wikiner import split_wikiner
import stanza.utils.datasets.ner.convert_bsf_to_beios as convert_bsf_to_beios
import stanza.utils.datasets.ner.convert_ijc as convert_ijc
import stanza.utils.datasets.ner.convert_rgai as convert_rgai
import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file

def process_turku(paths):
    short_name = 'fi_turku'
    base_input_path = os.path.join(paths["NERBASE"], short_name)
    base_output_path = paths["NER_DATA_DIR"]
    for shard in ('train', 'dev', 'test'):
        input_filename = os.path.join(base_input_path, '%s.tsv' % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def process_languk(paths):
    short_name = 'uk_languk'
    base_input_path = os.path.join(paths["NERBASE"], 'lang-uk', 'ner-uk', 'data')
    base_output_path = paths["NER_DATA_DIR"]
    convert_bsf_to_beios.convert_bsf_in_folder(base_input_path, base_output_path)
    for shard in ('train', 'dev', 'test'):
        input_filename = os.path.join(base_output_path, convert_bsf_to_beios.CORPUS_NAME, "%s.bio" % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(input_filename, output_filename)


def process_ijc(paths, short_name):
    """
    Splits the ijc Hindi dataset in train, dev, test

    The original data had train & test splits, so we randomly divide
    the files in train to make a dev set.

    The expected location of the IJC data is hi_ijc.  This method
    should be possible to use for other languages, but we have very
    little support for the other languages of IJC at the moment.
    """
    base_input_path = os.path.join(paths["NERBASE"], short_name)
    base_output_path = paths["NER_DATA_DIR"]

    test_files = [os.path.join(base_input_path, "test-data-hindi.txt")]
    test_csv_file = os.path.join(base_output_path, short_name + ".test.csv")
    print("Converting test input %s to space separated file in %s" % (test_files[0], test_csv_file))
    convert_ijc.convert_ijc(test_files, test_csv_file)

    train_input_path = os.path.join(base_input_path, "training-hindi", "*utf8")
    train_files = glob.glob(train_input_path)
    train_csv_file = os.path.join(base_output_path, short_name + ".train.csv")
    dev_csv_file = os.path.join(base_output_path, short_name + ".dev.csv")
    print("Converting training input from %s to space separated files in %s and %s" % (train_input_path, train_csv_file, dev_csv_file))
    convert_ijc.convert_split_ijc(train_files, train_csv_file, dev_csv_file)

    for csv_file, shard in zip((train_csv_file, dev_csv_file, test_csv_file), ("train", "dev", "test")):
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(csv_file, output_filename)


def process_fire_2013(paths, dataset):
    """
    Splits the FIRE 2013 dataset into train, dev, test

    The provided datasets are all mixed together at this point, so it
    is not possible to recreate the original test conditions used in
    the bakeoff
    """
    short_name = treebank_to_short_name(dataset)
    langcode, _ = short_name.split("_")
    if not langcode in ("hi", "en", "ta", "bn", "mal"):
        raise ValueError("Language %s not one of the FIRE 2013 languages")
    language = lcode2lang[langcode].lower()
    
    # for example, FIRE2013/hindi_train
    base_input_path = os.path.join(paths["NERBASE"], "FIRE2013", "%s_train" % language)
    base_output_path = paths["NER_DATA_DIR"]

    train_csv_file = os.path.join(base_output_path, "%s.train.csv" % short_name)
    dev_csv_file   = os.path.join(base_output_path, "%s.dev.csv" % short_name)
    test_csv_file  = os.path.join(base_output_path, "%s.test.csv" % short_name)

    convert_fire_2013(base_input_path, train_csv_file, dev_csv_file, test_csv_file)

    for csv_file, shard in zip((train_csv_file, dev_csv_file, test_csv_file), ("train", "dev", "test")):
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(csv_file, output_filename)

    
def process_wikiner(paths, dataset):
    short_name = treebank_to_short_name(dataset)

    base_input_path = os.path.join(paths["NERBASE"], dataset)
    base_output_path = paths["NER_DATA_DIR"]

    raw_input_path = os.path.join(base_input_path, "raw")
    input_files = glob.glob(os.path.join(raw_input_path, "aij-wikiner*"))
    if len(input_files) == 0:
        raise FileNotFoundError("Could not find any raw wikiner files in %s" % raw_input_path)
    elif len(input_files) > 1:
        raise FileNotFoundError("Found too many raw wikiner files in %s: %s" % (raw_input_path, ", ".join(input_files)))

    csv_file = os.path.join(raw_input_path, "csv_" + short_name)
    print("Converting raw input %s to space separated file in %s" % (input_files[0], csv_file))
    preprocess_wikiner(input_files[0], csv_file)

    # this should create train.bio, dev.bio, and test.bio
    print("Splitting %s to %s" % (csv_file, base_input_path))
    split_wikiner(base_input_path, csv_file)

    for shard in ('train', 'dev', 'test'):
        input_filename = os.path.join(base_input_path, '%s.bio' % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        print("Converting %s to %s" % (input_filename, output_filename))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def process_rgai(paths, short_name):
    base_output_path = paths["NER_DATA_DIR"]
    base_input_path = os.path.join(paths["NERBASE"], "hu_rgai")

    if short_name == 'hu_rgai':
        use_business = True
        use_criminal = True
    elif short_name == 'hu_rgai_business':
        use_business = True
        use_criminal = False
    elif short_name == 'hu_rgai_criminal':
        use_business = False
        use_criminal = True
    else:
        raise ValueError("Unknown subset of hu_rgai data: %s" % short_name)

    convert_rgai.convert_rgai(base_input_path, base_output_path, short_name, use_business, use_criminal)

    # TODO: refactor all these similar blocks
    for shard in ('train', 'dev', 'test'):
        input_filename = os.path.join(base_output_path, '%s.%s.bio' % (short_name, shard))
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        print("Converting %s to %s" % (input_filename, output_filename))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def process_nytk(paths, short_name):
    """
    Process the NYTK dataset
    TODO: include the rgai dataset as training data, at least as an experiment
    """
    base_output_path = paths["NER_DATA_DIR"]
    base_input_path = os.path.join(paths["NERBASE"], "NYTK-NerKor")

    for shard in ('train', 'dev', 'test'):
        if shard == 'dev':
            base_input_subdir = os.path.join(base_input_path, "data/train-devel-test/devel")
        else:
            base_input_subdir = os.path.join(base_input_path, "data/train-devel-test", shard)

        shard_lines = []
        subpaths = glob.glob(base_input_subdir + "/*/no-morph/*")
        for input_filename in subpaths:
            if len(shard_lines) > 0:
                shard_lines.append("")
            with open(input_filename) as fin:
                lines = fin.readlines()
                if lines[0].strip() != '# global.columns = FORM LEMMA UPOS XPOS FEATS CONLL:NER':
                    raise ValueError("Unexpected format in %s" % input_filename)
                lines = [x.strip().split("\t") for x in lines[1:]]
                lines = ["%s\t%s" % (x[0], x[5]) if len(x) > 1 else "" for x in lines]
                shard_lines.extend(lines)

        bio_filename = os.path.join(base_output_path, '%s.%s.bio' % (short_name, shard))
        with open(bio_filename, "w") as fout:
            for line in shard_lines:
                fout.write(line)
                fout.write("\n")

        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(bio_filename, output_filename)


def main():
    paths = default_paths.get_default_paths()

    dataset_name = sys.argv[1]
    random.seed(1234)

    if dataset_name == 'fi_turku':
        process_turku(paths)
    elif dataset_name in ('uk_languk', 'Ukranian_languk', 'Ukranian-languk'):
        process_languk(paths)
    elif dataset_name == 'hi_ijc':
        process_ijc(paths, dataset_name)
    elif dataset_name.endswith("FIRE2013"):
        process_fire_2013(paths, dataset_name)
    elif dataset_name.endswith('WikiNER'):
        process_wikiner(paths, dataset_name)
    elif dataset_name.startswith('hu_rgai'):
        process_rgai(paths, dataset_name)
    elif dataset_name.startswith('hu_nytk'):
        process_nytk(paths, dataset_name)
    else:
        raise ValueError(f"dataset {dataset_name} currently not handled")

if __name__ == '__main__':
    main()
