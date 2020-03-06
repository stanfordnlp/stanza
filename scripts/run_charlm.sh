#!/bin/bash
#
# Train and evaluate character-level language model. Run as:
#   ./run_charlm.sh CORPUS DIRECTION OTHER_ARGS
# where CORPUS is charlm corpus name (e.g., English-1Billion), DIRECTION is either forward or backward, and OTHER_ARGS are additional training arguments (see charlm code) or empty.
# This script assumes UDBASE and CHARLM_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

corpus=$1; shift
direction=$1; shift
args=$@

lang=`echo $corpus | sed -e 's#-.*$##g'`
lcode=`python scripts/lang2code.py $lang`
corpus_name=`echo $corpus | sed -e 's#^.*-##g' | tr '[:upper:]' '[:lower:]'`
short=${lcode}_${corpus_name}

train_dir=${CHARLM_DATA_DIR}/${lang}/${corpus_name}/train
dev_file=${CHARLM_DATA_DIR}/${lang}/${corpus_name}/dev.txt
test_file=${CHARLM_DATA_DIR}/${lang}/${corpus_name}/test.txt

echo "Running charlm for $lang:$corpus with $args..."
python -m stanza.models.charlm --train_dir $train_dir --eval_file $dev_file \
    --direction $direction --lang $lang --shorthand $short --mode train $args
python -m stanza.models.charlm --eval_file $dev_file \
    --direction $direction --lang $lang --shorthand $short --mode predict $args
python -m stanza.models.charlm --eval_file $test_file \
    --direction $direction --lang $lang --shorthand $short --mode predict $args
