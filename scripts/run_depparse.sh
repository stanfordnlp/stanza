#!/bin/bash
#
# Train and evaluate parser. Run as:
#   ./run_depparse.sh TREEBANK TAG_TYPE OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see parser code) or empty.
# This script assumes UDBASE and DEPPARSE_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
tag_type=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_file=${DEPPARSE_DATA_DIR}/${short}.train.in.conllu
eval_file=${DEPPARSE_DATA_DIR}/${short}.dev.in.conllu
output_file=${DEPPARSE_DATA_DIR}/${short}.dev.pred.conllu
gold_file=${DEPPARSE_DATA_DIR}/${short}.dev.gold.conllu

if [ ! -e $train_file ]; then
    bash scripts/prep_depparse_data.sh $treebank $tag_type
fi

# handle languages that need reduced batch size
batch_size=5000

if [ $treebank == 'UD_Finnish-TDT' ] || [ $treebank == 'UD_Russian-Taiga' ] || [ $treebank == 'UD_Latvian-LVTB' ] \
    || [ $treebank == 'UD_Croatian-SET' ] || [ $treebank == 'UD_Galician-TreeGal' ] || [ $treebank == 'UD_Czech-CLTT' ]; then
    batch_size=3000
elif [ $treebank == 'UD_German-HDT' ]; then
    batch_size=1500
fi

echo "Using batch size $batch_size"

echo "Running parser with $args..."
python -m stanfordnlp.models.parser --wordvec_dir $WORDVEC_DIR --train_file $train_file --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --batch_size $batch_size --mode train $args
python -m stanfordnlp.models.parser --wordvec_dir $WORDVEC_DIR --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode predict $args
results=`python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $output_file | head -12 | tail -n+12 | awk '{print $7}'`
echo $results $args >> ${DEPPARSE_DATA_DIR}/${short}.results
echo $short $results $args
