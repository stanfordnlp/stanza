CONLLBASE=/u/nlp/data/dependency_treebanks/CoNLL18/

for d in `ls $CONLLBASE`; do
    bn=`basename $d`
    short=`bash scripts/treebank_to_shorthand.sh ud $bn`
    train=$CONLLBASE$d/$short-ud-train.conllu

    if [[ $d != "UD_"* ]]; then
        continue
    fi

    if [[ $short = "_"* ]]; then
        echo "Couldn't identify language code for:" $d
        ls $CONLLBASE$d
        exit
    fi

    if [ -e $train ]; then
        tokens=`cat $train | grep -v "^#" | grep "\S" | wc -l`
        sentences=`cat $train | grep -v "^#" | grep -v "\S" | wc -l`
        echo $d" "$tokens" "$sentences
    else
        echo $d
    fi
done
