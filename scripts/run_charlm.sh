#!/bin/bash
#
# Train and evaluate character-level language model. Run as:
#   ./run_charlm.sh TREEBANK DIRECTION OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT), DIRECTION is either forward or backward, and OTHER_ARGS are additional training arguments (see charlm code) or empty.
# This script assumes UDBASE and CHARLM_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
direction=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_file=${CHARLM_DATA_DIR}/${short}.train.txt
dev_file=${CHARLM_DATA_DIR}/${short}.dev.txt
test_file=${CHARLM_DATA_DIR}/${short}.test.txt

echo "Running charlm with $args..."
python -m stanfordnlp.models.charlm --train_file $train_file --eval_file $dev_file \
    --direction $direction --lang $lang --shorthand $short --mode train $args
python -m stanfordnlp.models.charlm --eval_file $dev_file \
    --direction $direction --lang $lang --shorthand $short --mode predict $args
python -m stanfordnlp.models.charlm --eval_file $test_file \
    --direction $direction --lang $lang --shorthand $short --mode predict $args