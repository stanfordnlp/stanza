#!/bin/bash
#
# Train and evaluate lemmatizer. Run as:
#   ./run_lemma.sh TREEBANK OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see lemmatizer code) or empty.
# This script assumes UDBASE and LEMMA_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_file=${LEMMA_DATA_DIR}/${short}.train.in.conllu
eval_file=${LEMMA_DATA_DIR}/${short}.dev.in.conllu
output_file=${LEMMA_DATA_DIR}/${short}.dev.pred.conllu
gold_file=${LEMMA_DATA_DIR}/${short}.dev.gold.conllu

if [ ! -e $train_file ]; then
    bash scripts/prep_lemma_data.sh $treebank
fi

# handle languages that need less epochs
num_epoch=60
if [ $treebank == 'UD_Czech-PDT' ] || [ $treebank == 'UD_Russian-SynTagRus' ] || [ $treebank == 'UD_German-HDT' ]; then
    num_epoch=30
fi

echo "Running lemmatizer with $args..."
if [[ "$lang" == "vi" || "$lang" == "fro" ]]; then
    python -m stanza.models.identity_lemmatizer --data_dir $LEMMA_DATA_DIR --train_file $train_file --eval_file $eval_file \
        --output_file $output_file --gold_file $gold_file --lang $short
else
    python -m stanza.models.lemmatizer --data_dir $LEMMA_DATA_DIR --train_file $train_file --eval_file $eval_file \
        --output_file $output_file --gold_file $gold_file --lang $short --num_epoch $num_epoch --mode train $args
    python -m stanza.models.lemmatizer --data_dir $LEMMA_DATA_DIR --eval_file $eval_file \
        --output_file $output_file --gold_file $gold_file --lang $short --mode predict $args
fi
