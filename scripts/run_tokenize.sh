#!/bin/bash
#
# Train and evaluate tokenizer. Run as:
#   ./run_tokenize.sh TREEBANK OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see tokenizer code) or empty.
# This script assumes UDBASE and TOKENIZE_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

if [ $lang == "vi" ]; then
    labels=${TOKENIZE_DATA_DIR}/${short}-ud-train.json
    label_type=json_file
    eval_file="--json_file ${TOKENIZE_DATA_DIR}/${short}-ud-dev.json"
    train_eval_file="--dev_json_file ${TOKENIZE_DATA_DIR}/${short}-ud-dev.json"
else
    labels=${TOKENIZE_DATA_DIR}/${short}-ud-train.toklabels
    label_type=label_file
    eval_file="--txt_file ${TOKENIZE_DATA_DIR}/${short}.dev.txt"
    train_eval_file="--dev_txt_file ${TOKENIZE_DATA_DIR}/${short}.dev.txt --dev_label_file ${TOKENIZE_DATA_DIR}/${short}-ud-dev.toklabels"
    
    if [ $lang == "zh" ]; then
        args="$args --skip_newline"
    fi
fi

if [ ! -e $labels ]; then
    bash scripts/prep_tokenize_data.sh $treebank train
    bash scripts/prep_tokenize_data.sh $treebank dev
fi

DEV_GOLD=${TOKENIZE_DATA_DIR}/${short}.dev.gold.conllu
seqlen=$(python -c "from math import ceil; print(ceil($(python stanfordnlp/utils/avg_sent_len.py $labels) * 3 / 100) * 100)")

echo "Running tokenizer with $args..."
python -m stanfordnlp.models.tokenizer --${label_type} $labels --txt_file ${TOKENIZE_DATA_DIR}/${short}.train.txt --lang $lang --max_seqlen $seqlen --mwt_json_file ${TOKENIZE_DATA_DIR}/${short}-ud-dev-mwt.json $train_eval_file --dev_conll_gold $DEV_GOLD --conll_file ${TOKENIZE_DATA_DIR}/${short}.dev.pred.conllu --shorthand ${short} $args
python -m stanfordnlp.models.tokenizer --mode predict $eval_file --lang $lang --conll_file ${TOKENIZE_DATA_DIR}/${short}.dev.pred.conllu --shorthand $short --mwt_json_file ${TOKENIZE_DATA_DIR}/${short}-ud-dev-mwt.json $args

results=`python stanfordnlp/utils/conll18_ud_eval.py -v $DEV_GOLD ${TOKENIZE_DATA_DIR}/${short}.dev.pred.conllu | head -5 | tail -n+3 | awk '{print $7}' | pr --columns 3 -aJT`
echo $results $args >> ${TOKENIZE_DATA_DIR}/${short}.results
echo $short $results $args
