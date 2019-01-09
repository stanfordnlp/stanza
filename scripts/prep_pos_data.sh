#!/bin/bash
treebank=$1
shift
UDBASE=/u/nlp/data/dependency_treebanks/CoNLL18
UDPIPEBASE=$UDBASE/UDPipe_out
DATADIR=data/pos
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

#dev_conllu=$UDPIPEBASE/${short}-dev-pred-udpipe.conllu
train_in_file=$DATADIR/${short}.train.in.conllu
dev_in_file=$DATADIR/${short}.dev.in.conllu
dev_gold_file=$DATADIR/${short}.dev.gold.conllu
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
