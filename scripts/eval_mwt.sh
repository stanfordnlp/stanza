treebank=$1
shift
gpu=$1
shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`
args=$@
UDBASE=/u/nlp/data/dependency_treebanks/CoNLL18/
DATADIR=data/mwt

train_file=${short}.train.in.conllu
eval_file=${short}.dev.in.conllu
output_file=${short}.dev.pred.conllu
gold_file=$UDBASE/$treebank/${short}-ud-dev.conllu

if [ ! -e $DATADIR/$train_file ]; then
    bash scripts/prep_mwt_data.sh $treebank
fi

cp data/tokenize/${output_file} $DATADIR/${eval_file}

echo "Evaluating $args..."
CUDA_VISIBLE_DEVICES=$gpu python -m models.mwt_expander --data_dir $DATADIR --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $short --mode predict $args
#python utils/conll18_ud_eval.py -v $gold_file $DATADIR/$output_file | grep "Lemmas" | awk '{print $7}'
results=`python utils/conll18_ud_eval.py -v $gold_file data/mwt/$output_file | head -5 | tail -n+5 | awk '{print $7}'`
echo $results $args >> data/mwt/${short}.results
echo $short $results $args
