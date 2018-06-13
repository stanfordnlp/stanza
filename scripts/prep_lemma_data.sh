#!/bin/bash
treebank=$1; shift
data_dir=$1; shift
UDBASE=/u/nlp/data/dependency_treebanks/CoNLL18
UDPIPEBASE=$UDBASE/UDPipe_out
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_conllu=$UDBASE/$treebank/${short}-ud-train.conllu
train_in_file=$data_dir/${short}.train.in.conllu
dev_conllu=$UDPIPEBASE/${short}-dev-pred-udpipe.conllu
dev_in_file=$data_dir/${short}.dev.in.conllu
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

