#!/bin/bash
#
# Prepare data for training and evaluating NER taggers. Run as:
#   ./prep_ner_data.sh CORPUS
# where CORPUS is the full corpus name, with language as prefix (e.g., English-CoNLL03).
# This script assumes NER_DIR and NER_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

corpus=$1; shift
lang=`echo $corpus | sed -e 's#-.*$##g'`
lcode=`python scripts/lang2code.py $lang`
corpus_name=`echo $corpus | sed -e 's#^.*-##g' | tr '[:upper:]' '[:lower:]'`
short=${lcode}_${corpus_name}

train_file=$NERBASE/${corpus}/train.bio
dev_file=$NERBASE/${corpus}/dev.bio
test_file=$NERBASE/${corpus}/test.bio

train_json_file=$NER_DATA_DIR/${short}.train.json
dev_json_file=$NER_DATA_DIR/${short}.dev.json
test_json_file=$NER_DATA_DIR/${short}.test.json

# create json file if exists; otherwise create empty files
if [ -e $train_file ]; then
    python stanza/utils/datasets/ner/prepare_ner_file.py $train_file $train_json_file
else
    touch $train_json_file
fi
if [ -e $dev_file ]; then
    python stanza/utils/datasets/ner/prepare_ner_file.py $dev_file $dev_json_file
else
    touch $dev_json_file
fi
if [ -e $test_file ]; then
    python stanza/utils/datasets/ner/prepare_ner_file.py $test_file $test_json_file
else
    touch $test_json_file
fi

