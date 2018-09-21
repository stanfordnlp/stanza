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
DATADIR=data/pos

train_file=${DATADIR}/${short}.train.in.conllu
eval_file=${DATADIR}/${short}.dev.in.conllu
output_file=${DATADIR}/${short}.dev.${outputprefix}pred.conllu
gold_file=${DATADIR}/${short}.dev.gold.conllu

if [ ! -e $train_file ]; then
    bash scripts/prep_pos_data.sh $treebank
fi

total_lines=`grep -v '^#' $train_file | wc -l`
total_sents=`grep -v '\S' $train_file | wc -l`
batch_size=$((5000 * total_sents / (total_lines - total_sents)))

echo "Running $args..."
CUDA_VISIBLE_DEVICES=$gpu python -m models.tagger --train_file $train_file --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode train --save_dir ${outputprefix}saved_models/pos --batch_size $batch_size $args
CUDA_VISIBLE_DEVICES=$gpu python -m models.tagger --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode predict --save_dir ${outputprefix}saved_models/pos $args
results=`python utils/conll18_ud_eval.py -v $gold_file $output_file | head -9 | tail -n+9 | awk '{print $7}'`
echo $results $args >> ${DATADIR}/${short}.${outputprefix}results
echo $short $results $args
