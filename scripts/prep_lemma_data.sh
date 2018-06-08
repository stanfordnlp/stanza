#!/bin/bash
treebank=$1
shift
UDBASE=/u/nlp/data/dependency_treebanks/CoNLL18
UDPIPEBASE=$UDBASE/UDPipe_out
DATADIR=data/lemma
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

#cat $UDBASE/$treebank/${short}-ud-train.conllu | grep -vP "^#" | cut -d$'\t' -f2-3  > $DATADIR/${short}.train.lem
#cat $UDPIPEBASE/${short}-dev-pred-udpipe.conllu | grep -vP "^#" | cut -d$'\t' -f2  > $DATADIR/${short}.dev.lem

cp $UDBASE/$treebank/${short}-ud-train.conllu $DATADIR/${short}.train.in.conllu
cp $UDPIPEBASE/${short}-dev-pred-udpipe.conllu $DATADIR/${short}.dev.in.conllu
