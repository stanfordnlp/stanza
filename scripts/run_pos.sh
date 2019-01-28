#!/bin/bash
#
# Train and evaluate tagger. Run as:
#   ./run_pos.sh TREEBANK OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see tagger code) or empty.
# This script assumes UDBASE and POS_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_file=${POS_DATA_DIR}/${short}.train.in.conllu
eval_file=${POS_DATA_DIR}/${short}.dev.in.conllu
output_file=${POS_DATA_DIR}/${short}.dev.${outputprefix}pred.conllu
gold_file=${POS_DATA_DIR}/${short}.dev.gold.conllu

if [ ! -e $train_file ]; then
    bash scripts/prep_pos_data.sh $treebank
fi

# handle languages that need reduced batch size
batch_size=5000

if [ $treebank == 'UD_Croatian-SET' ]; then
    batch_size=3000
fi
echo "Using batch size $batch_size"

echo "Running tagger with $args..."
python -m stanfordnlp.models.tagger --wordvec_dir $WORDVEC_DIR --train_file $train_file --eval_file $eval_file \
    --output_file $output_file --batch_size $batch_size --gold_file $gold_file --lang $lang --shorthand $short \
    --mode train $args
python -m stanfordnlp.models.tagger --wordvec_dir $WORDVEC_DIR --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode predict $args
results=`python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $output_file | head -9 | tail -n+9 | awk '{print $7}'`
echo $results $args >> ${POS_DATA_DIR}/${short}.results
echo $short $results $args
