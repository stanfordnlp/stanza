outputprefix=$1
if [[ "$outputprefix" == "UD_"* ]]; then
    outputprefix=""
else
    shift
fi
treebank=$1
shift
gpu=$1
shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`
args=$@
UDBASE=/u/nlp/data/dependency_treebanks/CoNLL18/
DATADIR=data/mwt

train_file=${DATADIR}/${short}.train.in.conllu
eval_file=${DATADIR}/${short}.dev.in.conllu
output_file=${DATADIR}/${short}.dev.pred.conllu
gold_file=$UDBASE/$treebank/${short}-ud-dev.conllu

if [ ! -e $train_file ]; then
    bash scripts/prep_mwt_data.sh $treebank
fi

dec_len=$(python -c "from math import ceil; print(ceil($(python utils/max_mwt_length.py data/tokenize/${short}-ud-train-mwt.json data/tokenize/${short}-ud-dev-mwt.json) * 1.1 + 1))")

echo "Running $args..."
CUDA_VISIBLE_DEVICES=$gpu python -m models.mwt_expander --train_file $train_file --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode train --max_dec_len $dec_len --save_dir ${outputprefix}saved_models $args
CUDA_VISIBLE_DEVICES=$gpu python -m models.mwt_expander --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode predict --save_dir ${outputprefix}saved_models $args
results=`python utils/conll18_ud_eval.py -v $gold_file $output_file | head -5 | tail -n+5 | awk '{print $7}'`
echo $results $args >> ${DATADIR}/${short}.${outputprefix}results
echo $short $results $args
