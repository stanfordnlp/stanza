#!/bin/bash
#
# Train and evaluate MWT expander. Run as:
#   ./run_mwt.sh TREEBANK OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see mwt_expander code) or empty.
# This script assumes UDBASE and MWT_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_file=${MWT_DATA_DIR}/${short}.train.in.conllu
eval_file=${MWT_DATA_DIR}/${short}.dev.in.conllu
output_file=${MWT_DATA_DIR}/${short}.dev.${outputprefix}pred.conllu
gold_file=${MWT_DATA_DIR}/${short}.dev.gold.conllu

if [ ! -e $train_file ]; then
    bash scripts/prep_mwt_data.sh $treebank
fi

dec_len=$(python -c "from math import ceil; print(ceil($(python stanfordnlp/utils/max_mwt_length.py ${TOKENIZE_DATA_DIR}/${short}-ud-train-mwt.json ${TOKENIZE_DATA_DIR}/${short}-ud-dev-mwt.json) * 1.1 + 1))")

echo "Running $args..."
python -m stanfordnlp.models.mwt_expander --train_file $train_file --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode train --max_dec_len $dec_len $args
python -m stanfordnlp.models.mwt_expander --eval_file $eval_file \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode predict $args
results=`python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $output_file | head -5 | tail -n+5 | awk '{print $7}'`
echo $results $args >> ${MWT_DATA_DIR}/${short}.results
echo $short $results $args
