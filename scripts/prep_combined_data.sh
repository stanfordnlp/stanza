langprefix=$1
shift
UDBASE=/u/nlp/data/dependency_treebanks/CoNLL18

targetname=${langprefix}-Combined
targetshort=`bash scripts/treebank_to_shorthand.sh ud $targetname`
targetdir=$UDBASE/$targetname
if [ ! -e $targetdir ]; then
    mkdir -p $targetdir
    for tb in `ls -d $UDBASE/${langprefix}*`; do
        if [[ $tb = $targetdir ]]; then
            continue
        fi
        treebank=`basename $tb`
        short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
        for suffix in -ud-train.txt -ud-train.conllu -ud-dev.txt -ud-dev.conllu; do
            cat $tb/${short}${suffix} >> $targetdir/${targetshort}${suffix}
        done
    done
fi
