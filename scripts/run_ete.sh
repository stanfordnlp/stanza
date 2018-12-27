# set up config
source scripts/config.sh

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
bash scripts/prep_tokenize_data.sh $treebank ${set}
echo 'running tokenizer...'
CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.tokenizer --mode predict $eval_file --lang $model_lang --conll_file data/tokenize/${short}.${set}.${outputprefix}pred.ete.conllu --mwt_json_file data/tokenize/${short}-ud-${set}-mwt.json --shorthand $model_short $savedir $args
cp data/tokenize/${short}.${set}.${outputprefix}pred.ete.conllu $ete_file

# run the mwt expander
if [ -e saved_models/mwt/${short}_mwt_expander.pt ]; then
    echo 'running mwt expander...'
    CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.mwt_expander --mode predict --eval_file $ete_file --shorthand $model_short --output_file data/mwt/${short}.${set}.pred.ete.conllu --save_dir saved_models/mwt $args
    cp data/mwt/${short}.${set}.pred.ete.conllu $ete_file
fi

# run the part-of-speech tagger
echo 'running part-of-speech tagger...'
CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.tagger --eval_file $ete_file --output_file data/pos/${short}.${set}.${outputprefix}pred.ete.conllu --lang $model_short --shorthand $model_short --mode predict --save_dir saved_models/pos
cp data/pos/${short}.${set}.${outputprefix}pred.ete.conllu $ete_file

# run the lemmatizer
echo 'running lemmatizer...'
CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.lemmatizer --data_dir data/lemma --eval_file $ete_file --output_file data/lemma/${short}.${set}.${outputprefix}pred.ete.conllu --lang $model_short --mode predict 
cp data/lemma/${short}.${set}.${outputprefix}pred.ete.conllu $ete_file 

# handle languages that need reduced batch size
batch_size=5000

if [ $treebank == 'UD_Finnish-TDT' ] || [ $treebank == 'UD_Russian-Taiga' ] || [ $treebank == 'UD_Latvian-LVTB' ]; then
    batch_size=3000
fi

# run the dependency parser
echo 'running dependency parser...'
CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.parser --eval_file $ete_file --output_file data/depparse/${short}.${set}.${outputprefix}pred.ete.conllu --lang $model_short --shorthand $model_short --mode predict --batch_size $batch_size --save_dir saved_models/depparse
cp data/depparse/${short}.${set}.${outputprefix}pred.ete.conllu $ete_file

# get final output table
# copy over gold file
cp ${UDBASE}/${treebank}/${short}-ud-${set}.conllu data/ete
gold_file=data/ete/${short}-ud-${set}.conllu
# run official eval script
echo 'running official eval script'
# print out results
python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $ete_file
# store results to file
python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $ete_file > ${short}.ete.results
