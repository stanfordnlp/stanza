#!/bin/bash
#
# Prepare data for training and evaluating MWT expanders. Run as:
#   ./prep_mwt_data.sh TREEBANK
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT).
# This script assumes UDBASE and MWT_DATA_DIR are correctly set in config.sh.

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

train_conllu=$UDBASE/$src_treebank/${src_short}-ud-train.conllu
dev_conllu=$UDBASE/$src_treebank/${src_short}-ud-dev.conllu # gold dev
dev_gold_conllu=$UDBASE/$src_treebank/${src_short}-ud-dev.conllu

train_in_file=$MWT_DATA_DIR/${short}.train.in.conllu
dev_in_file=$MWT_DATA_DIR/${short}.dev.in.conllu
dev_gold_file=$MWT_DATA_DIR/${short}.dev.gold.conllu
# copy conllu file if exists; otherwise create empty files
if [ -e $train_conllu ]; then
    echo "Preparing training data..."
    cp $train_conllu $train_in_file
    bash scripts/prep_tokenize_data.sh $src_treebank train
else
    touch $train_in_file
fi

if [ -e $dev_conllu ]; then
    echo "Preparing dev data..."
    python stanfordnlp/utils/contract_mwt.py $dev_conllu $dev_in_file
    bash scripts/prep_tokenize_data.sh $src_treebank dev
else
    touch $dev_in_file
fi

if [ -e $dev_gold_conllu ]; then
    cp $dev_gold_conllu $dev_gold_file
else
    touch $dev_gold_file
fi
