#!/bin/bash
#
# Prepare data for training and evaluating taggers. Run as:
#   ./prep_pos_data.sh TREEBANK
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT).
# This script assumes UDBASE and POS_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift

if [ -z "$treebank" ]; then
    echo "No treebank argument provided.  Please run with ./prep_pos_data.sh TREEBANK"
    exit 1
fi

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

# ensure that the needed data directory exists
mkdir -p $POS_DATA_DIR

train_in_file=$POS_DATA_DIR/${short}.train.in.conllu
dev_in_file=$POS_DATA_DIR/${short}.dev.in.conllu
dev_gold_file=$POS_DATA_DIR/${short}.dev.gold.conllu

# copy conllu file if exists; otherwise create empty files
if [ -e $train_conllu ]; then
    cp $train_conllu $train_in_file
else
    touch $train_in_file
fi

if [ -e $dev_conllu ]; then
    cp $dev_conllu $dev_in_file
else
    touch $dev_in_file
fi

if [ -e $dev_gold_conllu ]; then
    cp $dev_gold_conllu $dev_gold_file
else
    touch $dev_gold_file
fi
