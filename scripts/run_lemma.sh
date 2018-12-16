source scripts/config.sh

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
DATADIR=data/lemma

train_file=${DATADIR}/${short}.train.in.conllu
eval_file=${DATADIR}/${short}.dev.in.conllu
output_file=${DATADIR}/${short}.dev.pred.conllu
gold_file=${DATADIR}/${short}.dev.gold.conllu

if [ ! -e $train_file ]; then
    bash scripts/prep_lemma_data.sh $treebank $DATADIR
fi

echo "Running $args..."
if [[ "$lang" == "vi" || "$lang" == "fro" ]]; then
    python -m stanfordnlp.models.identity_lemmatizer --data_dir $DATADIR --train_file $train_file --eval_file $eval_file \
        --output_file $output_file --gold_file $gold_file --lang $short
else
    CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.lemmatizer --data_dir $DATADIR --train_file $train_file --eval_file $eval_file \
        --output_file $output_file --gold_file $gold_file --lang $short --mode train $args
    CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.lemmatizer --data_dir $DATADIR --eval_file $eval_file \
        --output_file $output_file --gold_file $gold_file --lang $short --mode predict
fi
#python utils/conll18_ud_eval.py -v $gold_file $DATADIR/$output_file | grep "Lemmas" | awk '{print $7}'
