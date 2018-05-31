treebank=$1
shift
set=$1
shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`
UDBASE=/u/nlp/data/dependency_treebanks/CoNLL18
python utils/prepare_tokenizer_data.py $UDBASE/$treebank/${short}-ud-train.txt $UDBASE/$treebank/${short}-ud-train.conllu -o data/${short}-ud-${set}.toklabels -m data/${short}-ud-${set}-mwt.json
if [ $lang == "vi" ]; then
    python utils/postprocess_vietnamese_tokenizer_data.py $UDBASE/$treebank/${short}-ud-${set}.txt data/${short}-ud-${set}.toklabels -o data/${short}-ud-${set}.json
fi
