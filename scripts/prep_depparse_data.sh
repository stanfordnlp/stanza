#!/bin/bash
#
# Prepare data for training and evaluating parsers. Run as:
#   ./prep_depparse_data.sh TREEBANK TAG_TYPE
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and TAG_TYPE is one of gold or predicted.
# This script assumes UDBASE and DEPPARSE_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

set -e

if hash python3 2>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

treebank=$1; shift
tag_type=$1; shift

original_short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

if [ -d "$UDBASE/${treebank}_XV" ]; then
    src_treebank="${treebank}_XV"
    src_short="${original_short}_xv"
else
    src_treebank=$treebank
    src_short=$original_short
fi

# path of input data to dependency parser training process
train_in_file=$DEPPARSE_DATA_DIR/${original_short}.train.in.conllu
dev_in_file=$DEPPARSE_DATA_DIR/${original_short}.dev.in.conllu
dev_gold_file=$DEPPARSE_DATA_DIR/${original_short}.dev.gold.conllu

# ensure that the needed data directory exists
mkdir -p $DEPPARSE_DATA_DIR

# handle languages requiring special batch size
batch_size=5000

if [ $treebank == 'UD_Galician-TreeGal' ]; then
    batch_size=3000
fi
echo "Using batch size $batch_size"

if [ -z "$tag_type" ]; then
    echo "Please specify either gold or predicted for tag type"
elif [ $tag_type == 'gold' ]; then
    train_conllu=$UDBASE/$src_treebank/${src_short}-ud-train.conllu
    dev_conllu=$UDBASE/$src_treebank/${src_short}-ud-dev.conllu # gold dev
    dev_gold_conllu=$UDBASE/$src_treebank/${src_short}-ud-dev.conllu
    cp $train_conllu $train_in_file
    cp $dev_conllu $dev_in_file
    cp $dev_gold_conllu $dev_gold_file
elif [ $tag_type == 'predicted' ]; then
    # build predicted tags
    # this assumes the part-of-speech tagging model has been built
    gold_train_file=$UDBASE/$src_treebank/${src_short}-ud-train.conllu
    gold_dev_file=$UDBASE/$src_treebank/${src_short}-ud-dev.conllu
    # run part-of-speech tagging on the train file
    echo '---'
    echo 'running part of speech model to generate predicted tags for train data'
    train_cmd='$PYTHON -m stanza.models.tagger --wordvec_dir '${WORDVEC_DIR}' --eval_file '${gold_train_file}' --gold_file '${gold_train_file}' --output_file '${train_in_file}' --lang '${original_short}' --shorthand '${original_short}' --batch_size '${batch_size}' --mode predict'
    echo ''
    echo $train_cmd
    echo ''
    eval $train_cmd
    # run part-of-speech tagging on the train file
    echo '---'
    echo 'running part of speech model to generate predicted tags for dev data'
    dev_cmd='$PYTHON -m stanza.models.tagger --wordvec_dir '${WORDVEC_DIR}' --eval_file '${gold_dev_file}' --gold_file '${gold_dev_file}' --output_file '${dev_in_file}' --lang '${original_short}' --shorthand '${original_short}' --batch_size '${batch_size}' --mode predict'
    echo ''
    echo $dev_cmd
    eval $dev_cmd
    cp $dev_in_file $dev_gold_file
else
    echo "Please specify either gold or predicted for tag type"
fi
