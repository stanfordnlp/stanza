treebank=$1
shift
gpu=$1
shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`
args=$@
UDBASE=/u/nlp/data/dependency_treebanks/CoNLL18/
DATADIR=data/lemma

train_file=${short}.train.lem
eval_file=${short}.dev.lem
output_file=${short}.dev.pred.conllu
gold_file=$UDBASE/$treebank/${short}-ud-dev.conllu

if [ ! -e $DATADIR/$train_file ]; then
    bash scripts/prep_lemma_data.sh $treebank
fi

#seqlen=$(python -c "from math import ceil; print(ceil($(python utils/avg_sent_len.py $labels) * 3 / 100) * 100)")
echo "Running $args..."
CUDA_VISIBLE_DEVICES=$gpu python -m models.lemmatizer --data_dir $DATADIR --train_file $train_file --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --mode train $args
#CUDA_VISIBLE_DEVICES=$gpu python -m models.tokenizer --mode predict $eval_file --lang $lang --mwt_json_file data/${short}-ud-train-mwt.json --conll_file ${short}-dev-pred.conllu $args
#results=`python utils/conll18_ud_eval.py -v /u/nlp/data/dependency_treebanks/CoNLL18/${treebank}/${short}-ud-dev.conllu ${short}-dev-pred.conllu | head -5 | tail -n+3 | awk '{print $7}' | pr --columns 3 -aJT`
#echo $results $args >> data/${short}.results
#echo $short $results $args
