# set up config
source scripts/config.sh

# show machine name
echo '---'
echo 'running full end to end pipeline'
machine_name=$(eval 'hostname')
echo 'running on: '${machine_name}

# get command line arguments
outputprefix=$1
if [[ "$outputprefix" == "UD_"* ]]; then
    outputprefix=""
else
    shift
fi
treebank=$1
shift
gpu=$1
shift
set=$1
shift

# set up short and lang
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
if [[ "$short" == *"_xv" ]]; then
    short=`echo $short | rev | cut -d_ -f1- | rev`
fi
lang=`echo $short | sed -e 's#_.*##g'`
args=$@

# set up savedir
if [[ "$args" == *"--save_dir"* ]]; then
    savedir=""
else
    savedir="--save_dir ${outputprefix}saved_models/tokenize"
fi

# copy initial text data
if [ ! -e data/tokenize/${short}-ud-${set}.txt ]; then
    echo 'copying test data for: '${treebank}
    cp ${UDBASE}/${treebank}/${short}-ud-${set}.txt data/tokenize
fi

# location of ete file to update
ete_file=data/ete/${short}.${set}.pred.ete.conllu

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
    eval_file="--json_file data/tokenize/${short}-ud-${set}.json"
else
    eval_file="--txt_file data/tokenize/${short}-ud-${set}.txt"
fi

# prep the dev/test data
echo '---'
echo 'running tokenizer...'
prep_tokenize_cmd="bash scripts/prep_tokenize_data.sh ${treebank} ${set}"
echo 'prepare tokenize data'
echo $prep_tokenize_cmd
eval $prep_tokenize_cmd

run_tokenize_cmd="CUDA_VISIBLE_DEVICES=${gpu} python -m models.tokenizer --mode predict ${eval_file} --lang ${model_lang} --conll_file data/tokenize/${short}.${set}.${outputprefix}pred.ete.conllu --shorthand ${model_short} ${savedir} ${args}"

echo 'run tokenizer'
echo $run_tokenize_cmd
eval $run_tokenize_cmd

echo 'update full ete file'

cp_ete_file_cmd="cp data/tokenize/${short}.${set}.${outputprefix}pred.ete.conllu ${ete_file}"

echo $cp_ete_file_cmd
eval $cp_ete_file_cmd

# run the mwt expander
if [ -e saved_models/mwt/${short}_mwt_expander.pt ]; then
    echo '---'
    echo 'running mwt expander...'
    run_mwt_cmd="CUDA_VISIBLE_DEVICES=${gpu} python -m models.mwt_expander --mode predict --eval_file ${ete_file} --shorthand ${model_short} --output_file data/mwt/${short}.${set}.pred.ete.conllu --save_dir saved_models/mwt ${args}"
    echo 'run mwt expander'
    echo $run_mwt_cmd
    eval $run_mwt_cmd
    echo 'update full ete file'
    cp_ete_file_cmd="cp data/mwt/${short}.${set}.pred.ete.conllu ${ete_file}"
    echo $cp_ete_file_cmd
    eval $cp_ete_file_cmd
fi

# run the part-of-speech tagger
echo '---'
echo 'running part-of-speech tagger...'
part_of_speech_cmd="CUDA_VISIBLE_DEVICES=${gpu} python -m models.tagger --eval_file ${ete_file} --output_file data/pos/${short}.${set}.${outputprefix}pred.ete.conllu --lang ${model_short} --shorthand ${model_short} --mode predict --save_dir saved_models/pos"
echo 'run part-of-speech'
echo $part_of_speech_cmd
eval $part_of_speech_cmd
echo 'update full ete file'
cp_ete_file_cmd="cp data/pos/${short}.${set}.${outputprefix}pred.ete.conllu $ete_file"
echo $cp_ete_file_cmd
eval $cp_ete_file_cmd

# run the lemmatizer
echo '---'
echo 'running lemmatizer...'
if [[ "$lang" == "vi" || "$lang" == "fro" ]]; then
    lemma_cmd="python -m models.identity_lemmatizer --data_dir data/lemma --eval_file ${ete_file} --output_file data/lemma/${short}.${set}.${outputprefix}pred.ete.conllu --lang ${model_short} --mode predict"
else
    lemma_cmd="CUDA_VISIBLE_DEVICES=${gpu} python -m models.lemmatizer --data_dir data/lemma --eval_file ${ete_file} --output_file data/lemma/${short}.${set}.${outputprefix}pred.ete.conllu --lang ${model_short} --mode predict"
fi
echo 'run lemmatizer'
echo $lemma_cmd
eval $lemma_cmd
echo 'update full ete file'
cp_ete_file_cmd="cp data/lemma/${short}.${set}.${outputprefix}pred.ete.conllu ${ete_file}"
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
depparse_cmd="CUDA_VISIBLE_DEVICES=${gpu} python -m models.parser --eval_file ${ete_file} --output_file data/depparse/${short}.${set}.${outputprefix}pred.ete.conllu --lang ${model_short} --shorthand ${model_short} --mode predict --batch_size ${batch_size} --save_dir saved_models/depparse"
echo 'run dependency parser'
echo $depparse_cmd
eval $depparse_cmd
cp_ete_file_cmd="cp data/depparse/${short}.${set}.${outputprefix}pred.ete.conllu ${ete_file}"
echo 'update full ete file'
echo $cp_ete_file_cmd
eval $cp_ete_file_cmd

# get final output table
# copy over gold file
cp ${UDBASE}/${treebank}/${short}-ud-${set}.conllu data/ete
gold_file=data/ete/${short}-ud-${set}.conllu
# run official eval script
echo 'running official eval script'
# print out results
python utils/conll18_ud_eval.py -v $gold_file $ete_file
# store results to file
python utils/conll18_ud_eval.py -v $gold_file $ete_file > ${short}.ete.results
