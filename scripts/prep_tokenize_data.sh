treebank=$1
shift
set=$1
shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`
UDBASE=/u/nlp/data/dependency_treebanks/CoNLL18
python utils/prepare_tokenizer_data.py $UDBASE/$treebank/${short}-ud-${set}.txt $UDBASE/$treebank/${short}-ud-${set}.conllu -o data/tokenize/${short}-ud-${set}.toklabels -m data/tokenize/${short}-ud-${set}-mwt.json
if [ $lang == "vi" ]; then
    python utils/postprocess_vietnamese_tokenizer_data.py $UDBASE/$treebank/${short}-ud-${set}.txt --char_level_pred data/tokenize/${short}-ud-${set}.toklabels -o data/tokenize/${short}-ud-${set}.json
fi
