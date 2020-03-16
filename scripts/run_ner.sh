#!/bin/bash
#
# Train and evaluate NER tagger. Run as:
#   ./run_ner.sh CORPUS OTHER_ARGS
# where CORPUS is the full corpus name (e.g., English-CoNLL03) and OTHER_ARGS are additional training arguments (see tagger code) or empty.
# This script assumes UDBASE and NER_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

corpus=$1; shift
args=$@

lang=`echo $corpus | sed -e 's#-.*$##g'`
lcode=`python scripts/lang2code.py $lang`
corpus_name=`echo $corpus | sed -e 's#^.*-##g' | tr '[:upper:]' '[:lower:]'`
short=${lcode}_${corpus_name}

train_file=${NER_DATA_DIR}/${short}.train.json
dev_file=${NER_DATA_DIR}/${short}.dev.json
test_file=${NER_DATA_DIR}/${short}.test.json

if [ ! -e $train_file ]; then
    bash scripts/prep_ner_data.sh $corpus
fi

echo "Running ner with $args..."
python -m stanza.models.ner_tagger --wordvec_dir $WORDVEC_DIR --train_file $train_file --eval_file $dev_file \
    --lang $lang --shorthand $short --mode train $args
python -m stanza.models.ner_tagger --wordvec_dir $WORDVEC_DIR --eval_file $dev_file \
    --lang $lang --shorthand $short --mode predict $args
python -m stanza.models.ner_tagger --wordvec_dir $WORDVEC_DIR --eval_file $test_file \
    --lang $lang --shorthand $short --mode predict $args

