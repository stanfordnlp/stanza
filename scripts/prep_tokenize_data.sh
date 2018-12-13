source scripts/config.sh

treebank=$1
shift
set=$1
shift

if [ -d $UDBASE/${treebank}_XV ]; then
    treebank=${treebank}_XV
fi

short=`bash scripts/treebank_to_shorthand.sh ud $treebank`

if [[ "$short" == *"_xv" ]]; then
    short1=`echo $short | rev | cut -d_ -f2- | rev`
else
    short1=$short
fi

lang=`echo $short | sed -e 's#_.*##g'`
python stanfordnlp/utils/prepare_tokenizer_data.py $UDBASE/$treebank/${short}-ud-${set}.txt $UDBASE/$treebank/${short}-ud-${set}.conllu -o data/tokenize/${short1}-ud-${set}.toklabels -m data/tokenize/${short1}-ud-${set}-mwt.json
cp $UDBASE/$treebank/${short}-ud-${set}.conllu data/tokenize/${short1}.${set}.gold.conllu
cp $UDBASE/$treebank/${short}-ud-${set}.txt data/tokenize/${short1}.${set}.txt
if [ $lang == "vi" ]; then
    python stanfordnlp/utils/postprocess_vietnamese_tokenizer_data.py $UDBASE/$treebank/${short}-ud-${set}.txt --char_level_pred data/tokenize/${short1}-ud-${set}.toklabels -o data/tokenize/${short1}-ud-${set}.json
fi
