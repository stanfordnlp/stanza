#!/bin/bash
#
# Prepare data for training and evaluating tokenizers. Run as:
#   ./prep_tokenize_data.sh TREEBANK DATASET
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and DATASET is one of train, dev or test.
# This script assumes TOKENIZE_DATA_DIR is correctly set in config.sh.
# UDBASE needs to be set externally

if hash python3 2>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

if [ -z "$TOKENIZE_DATA_DIR" ]; then
    source scripts/config.sh
fi

treebank=$1; shift
dataset=$1; shift

#if [ -d $UDBASE/${treebank}_XV ] && [ ! $dataset=='test' ]; then
#    treebank=${treebank}_XV
#fi

short=`bash scripts/treebank_to_shorthand.sh ud $treebank`

#if [[ "$short" == *"_xv" ]]; then
#    short1=`echo $short | rev | cut -d_ -f2- | rev`
#else
#    short1=$short
#fi

lang=`echo $short | sed -e 's#_.*##g'`
echo "Preparing tokenizer $dataset data..."
$PYTHON stanza/utils/prepare_tokenizer_data.py $UDBASE/$treebank/${short}-ud-${dataset}.txt $UDBASE/$treebank/${short}-ud-${dataset}.conllu -o ${TOKENIZE_DATA_DIR}/${short}-ud-${dataset}.toklabels -m ${TOKENIZE_DATA_DIR}/${short}-ud-${dataset}-mwt.json
cp $UDBASE/$treebank/${short}-ud-${dataset}.conllu ${TOKENIZE_DATA_DIR}/${short}.${dataset}.gold.conllu
cp $UDBASE/$treebank/${short}-ud-${dataset}.txt ${TOKENIZE_DATA_DIR}/${short}.${dataset}.txt
# handle Vietnamese data
if [ $lang == "vi" ]; then
    python stanza/utils/postprocess_vietnamese_tokenizer_data.py $UDBASE/$treebank/${short}-ud-${dataset}.txt --char_level_pred ${TOKENIZE_DATA_DIR}/${short}-ud-${dataset}.toklabels -o ${TOKENIZE_DATA_DIR}/${short}-ud-${dataset}.json
fi
