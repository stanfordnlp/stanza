#!/bin/bash
#
# Prepare data for training and evaluating NER taggers. Run as:
#   ./prep_ner_data.sh TREEBANK
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT).
# This script assumes NER_DIR and NER_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_file=$NER_DIR/${treebank}/${lang}.train
dev_file=$NER_DIR/${treebank}/${lang}.testa
test_file=$NER_DIR/${treebank}/${lang}.testb

train_json_file=$NER_DATA_DIR/${short}.train.json
dev_json_file=$NER_DATA_DIR/${short}.dev.json
test_json_file=$NER_DATA_DIR/${short}.test.json

# create json file if exists; otherwise create empty files
if [ -e $train_file ]; then
    python stanfordnlp/utils/prepare_ner_data.py $train_file $train_json_file
else
    touch $train_json_file
fi
if [ -e $dev_file ]; then
    python stanfordnlp/utils/prepare_ner_data.py $dev_file $dev_json_file
else
    touch $dev_json_file
fi
if [ -e $test_file ]; then
    python stanfordnlp/utils/prepare_ner_data.py $test_file $test_json_file
else
    touch $test_json_file
fi

