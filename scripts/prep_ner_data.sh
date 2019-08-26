#!/bin/bash
#
# Prepare data for training and evaluating NER taggers. Run as:
#   ./prep_ner_data.sh TREEBANK
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT).
# This script assumes UDBASE and NER_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

if [ -d "$UDBASE/${treebank}_XV" ]; then
    src_treebank="${treebank}_XV"
    src_short="${short}_xv"
else
    src_treebank=$treebank
    src_short=$short
fi

train_file=$NER_DIR/${lang}.train
dev_file=$NER_DIR/${lang}.testa
test_file=$NER_DIR/${lang}.testb

train_conllu_file=$NER_DATA_DIR/${lang}.train.in.conllu
dev_conllu_file=$NER_DATA_DIR/${lang}.dev.in.conllu
test_conllu_file=$NER_DATA_DIR/${lang}.test.in.conllu

# create conllu file if exists; otherwise create empty files
if [ -e $train_file ]; then
    python stanfordnlp/utils/prepare_ner_data.py $train_file $train_conllu_file
else
    touch $train_conllu_file
fi
if [ -e $dev_file ]; then
    python stanfordnlp/utils/prepare_ner_data.py $dev_file $dev_conllu_file
else
    touch $dev_conllu_file
fi
if [ -e $test_file ]; then
    python stanfordnlp/utils/prepare_ner_data.py $test_file $test_conllu_file
else
    touch $test_conllu_file
fi

