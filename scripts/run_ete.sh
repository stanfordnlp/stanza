#/bin/bash
#
# Run an end-to-end evaluation of all trained modules. Run as:
#   ./run_ete.sh TREEBANK SET
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and set is one of train, dev or test.
# This script assumes environment variables are correctly set is config.sh.

# set up config
source scripts/config.sh

# show machine name
echo '---'
echo 'running full end to end pipeline'
machine_name=$(eval 'hostname')
echo 'running on: '${machine_name}

# get command line arguments
treebank=$1; shift
set=$1; shift

# set up short and lang
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
if [[ "$short" == *"_xv" ]]; then
    short=`echo $short | rev | cut -d_ -f1- | rev`
fi
lang=`echo $short | sed -e 's#_.*##g'`

# copy initial text data
if [ ! -e ${TOKENIZE_DATA_DIR}/${short}-ud-${set}.txt ]; then
    echo 'copying test data for: '${treebank}
    cp ${UDBASE}/${treebank}/${short}-ud-${set}.txt ${TOKENIZE_DATA_DIR}
fi

# location of ete file to update
ete_file=${ETE_DATA_DIR}/${short}.${set}.pred.ete.conllu

# if necessary select backoff model
model_short=$short
model_lang=$lang

if [ ! -e saved_models/tokenize/${short}_tokenizer.pt ]; then
    model_short=`python stanfordnlp/utils/select_backoff.py $treebank`
    model_lang=$model_short
    echo 'using backoff model'
    echo $treebank' --> '$model_short
fi

# run the tokenizer
# variables for tokenizer
if [ $lang == "vi" ]; then
    eval_file="--json_file ${TOKENIZE_DATA_DIR}/${short}-ud-${set}.json"
else
    eval_file="--txt_file ${TOKENIZE_DATA_DIR}/${short}-ud-${set}.txt"
fi

# prep the dev/test data
echo '---'
echo 'running tokenizer...'
prep_tokenize_cmd="bash scripts/prep_tokenize_data.sh ${treebank} ${set}"
echo 'prepare tokenize data'
echo $prep_tokenize_cmd
eval $prep_tokenize_cmd

run_tokenize_cmd="python -m stanfordnlp.models.tokenizer --mode predict ${eval_file} --lang ${model_lang} --conll_file ${TOKENIZE_DATA_DIR}/${short}.${set}.pred.ete.conllu --shorthand ${model_short}"

echo 'run tokenizer'
echo $run_tokenize_cmd
eval $run_tokenize_cmd

echo 'update full ete file'

cp_ete_file_cmd="cp ${TOKENIZE_DATA_DIR}/${short}.${set}.pred.ete.conllu ${ete_file}"

echo $cp_ete_file_cmd
eval $cp_ete_file_cmd

# run the mwt expander
if [ -e saved_models/mwt/${short}_mwt_expander.pt ]; then
    echo '---'
    echo 'running mwt expander...'
    run_mwt_cmd="python -m stanfordnlp.models.mwt_expander --mode predict --eval_file ${ete_file} --shorthand ${model_short} --output_file ${MWT_DATA_DIR}/${short}.${set}.pred.ete.conllu"
    echo 'run mwt expander'
    echo $run_mwt_cmd
    eval $run_mwt_cmd
    echo 'update full ete file'
    cp_ete_file_cmd="cp ${MWT_DATA_DIR}/${short}.${set}.pred.ete.conllu ${ete_file}"
    echo $cp_ete_file_cmd
    eval $cp_ete_file_cmd
fi

# run the part-of-speech tagger
echo '---'
echo 'running part-of-speech tagger...'
part_of_speech_cmd="python -m stanfordnlp.models.tagger --wordvec_dir ${WORDVEC_DIR} --eval_file ${ete_file} --output_file ${POS_DATA_DIR}/${short}.${set}.pred.ete.conllu --lang ${model_short} --shorthand ${model_short} --mode predict"
echo 'run part-of-speech'
echo $part_of_speech_cmd
eval $part_of_speech_cmd
echo 'update full ete file'
cp_ete_file_cmd="cp ${POS_DATA_DIR}/${short}.${set}.pred.ete.conllu $ete_file"
echo $cp_ete_file_cmd
eval $cp_ete_file_cmd

# run the lemmatizer
echo '---'
echo 'running lemmatizer...'
if [[ "$lang" == "vi" || "$lang" == "fro" ]]; then
    lemma_cmd="python -m stanfordnlp.models.identity_lemmatizer --data_dir ${LEMMA_DATA_DIR} --eval_file ${ete_file} --output_file ${LEMMA_DATA_DIR}/${short}.${set}.pred.ete.conllu --lang ${model_short} --mode predict"
else
    lemma_cmd="python -m stanfordnlp.models.lemmatizer --data_dir ${LEMMA_DATA_DIR} --eval_file ${ete_file} --output_file ${LEMMA_DATA_DIR}/${short}.${set}.pred.ete.conllu --lang ${model_short} --mode predict"
fi
echo 'run lemmatizer'
echo $lemma_cmd
eval $lemma_cmd
echo 'update full ete file'
cp_ete_file_cmd="cp ${LEMMA_DATA_DIR}/${short}.${set}.pred.ete.conllu ${ete_file}"
echo $cp_ete_file_cmd
eval $cp_ete_file_cmd

# handle languages that need reduced batch size
batch_size=5000

if [ $treebank == 'UD_Finnish-TDT' ] || [ $treebank == 'UD_Russian-Taiga' ] || [ $treebank == 'UD_Latvian-LVTB' ]; then
    batch_size=3000
fi

# run the dependency parser
echo '---'
echo 'running dependency parser...'
depparse_cmd="python -m stanfordnlp.models.parser --wordvec_dir ${WORDVEC_DIR} --eval_file ${ete_file} --output_file ${DEPPARSE_DATA_DIR}/${short}.${set}.pred.ete.conllu --lang ${model_short} --shorthand ${model_short} --mode predict --batch_size ${batch_size}"
echo 'run dependency parser'
echo $depparse_cmd
eval $depparse_cmd
cp_ete_file_cmd="cp ${DEPPARSE_DATA_DIR}/${short}.${set}.pred.ete.conllu ${ete_file}"
echo 'update full ete file'
echo $cp_ete_file_cmd
eval $cp_ete_file_cmd

# get final output table
# copy over gold file
cp ${UDBASE}/${treebank}/${short}-ud-${set}.conllu ${ETE_DATA_DIR}
gold_file=${ETE_DATA_DIR}/${short}-ud-${set}.conllu
# run official eval script
echo 'running official eval script'
# print out results
python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $ete_file
# store results to file
python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $ete_file > ${short}.ete.results
