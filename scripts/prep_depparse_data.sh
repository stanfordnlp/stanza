#!/bin/bash
source scripts/config.sh
treebank=$1
shift
tag_type=$1
shift
UDPIPEBASE=$UDBASE/UDPipe_out
DATADIR=data/depparse
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
train_in_file=$DATADIR/${original_short}.train.in.conllu
dev_in_file=$DATADIR/${original_short}.dev.in.conllu
dev_gold_file=$DATADIR/${original_short}.dev.gold.conllu

# handle languages requiring special batch size
batch_size=5000

if [ $treebank == 'UD_Galician-TreeGal' ]; then
    batch_size=3000
fi

if [ $tag_type == 'gold' ]; then
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
    train_cmd='python -m stanfordnlp.models.tagger --eval_file '${gold_train_file}' --gold_file '${gold_train_file}' --output_file '${train_in_file}' --lang '${original_short}' --shorthand '${original_short}' --batch_size '${batch_size}' --mode predict --save_dir saved_models/pos'
    echo ''
    echo $train_cmd
    echo ''
    eval $train_cmd
    # run part-of-speech tagging on the train file
    echo '---'
    echo 'running part of speech model to generate predicted tags for dev data'
    dev_cmd='python -m stanfordnlp.models.tagger --eval_file '${gold_dev_file}' --gold_file '${gold_dev_file}' --output_file '${dev_in_file}' --lang '${original_short}' --shorthand '${original_short}' --batch_size '${batch_size}' --mode predict --save_dir saved_models/pos'
    echo ''
    echo $dev_cmd
    eval $dev_cmd
    cp $dev_in_file $dev_gold_file
fi
