#!/bin/bash
#
# Train and evaluate NER tagger. Run as:
#   ./run_ner.sh TREEBANK OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see tagger code) or empty.
# This script assumes UDBASE and NER_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_file=${NER_DATA_DIR}/${lang}.train.json
dev_file=${NER_DATA_DIR}/${lang}.dev.json
test_file=${NER_DATA_DIR}/${lang}.test.json

if [ ! -e $train_file ]; then
    bash scripts/prep_ner_data.sh $treebank
fi

echo "Running ner with $args..."
python -m stanfordnlp.models.ner_tagger --wordvec_dir $WORDVEC_DIR --train_file $train_file --eval_file $dev_file \
    --lang $lang --shorthand $short --mode train $args
python -m stanfordnlp.models.ner_tagger --wordvec_dir $WORDVEC_DIR --eval_file $dev_file \
    --lang $lang --shorthand $short --mode predict $args
python -m stanfordnlp.models.ner_tagger --wordvec_dir $WORDVEC_DIR --eval_file $test_file \
    --lang $lang --shorthand $short --mode predict $args

