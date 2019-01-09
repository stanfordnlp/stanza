CONLLBASE=/u/nlp/data/dependency_treebanks/CoNLL18/
UDPIPE=udpipe-1.2.0-bin/bin-linux64/udpipe
UDPIPE_MODELS=udpipe-1.2.0-bin/models/
EVAL="python utils/conll18_ud_eval.py -v"

for d in `ls $CONLLBASE`; do
    bn=`basename $d`
    short=`bash scripts/treebank_to_shorthand.sh ud $bn`
    short1=`echo $d | cut -d"_" -f2- | tr [:upper:] [:lower:]`
    modelname=$UDPIPE_MODELS${short1}-ud-2.2-conll18-180430.udpipe
    gold=$CONLLBASE$d/$short-ud-dev.conllu
    pred=data/$short-dev-pred-udpipe.conllu

    if [[ $d != "UD_"* ]]; then
        continue
    fi

    if [ -e $gold ]; then
        $UDPIPE --tokenize --tag --parse $modelname $CONLLBASE$d/$short-ud-dev.txt --outfile=$pred 2>/dev/null 1>/dev/null
        echo $d" "`$EVAL $gold $pred | tail -n+3 | awk '{print $7}' | pr --columns 13 -aTJ` 2>/dev/null
    else
        echo $d
    fi
done
