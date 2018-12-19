# set up config
source scripts/config.sh


# get command line parameters
outputprefix=$1
if [[ "$outputprefix" == "UD_"* ]]; then
    outputprefix=""
else
    shift
fi
treebank=$1
shift
mode=$1
shift
gpu=$1
shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
if [[ "$short" == *"_xv" ]]; then
    short=`echo $short | rev | cut -d_ -f1- | rev`
fi
lang=`echo $short | sed -e 's#_.*##g'`
args=$@
if [ $lang == "vi" ]; then
    labels=data/tokenize/${short}-ud-train.json
    label_type=json_file
    eval_file="--json_file data/tokenize/${short}-ud-dev.json"
    train_eval_file="--dev_json_file data/tokenize/${short}-ud-dev.json"
else
    labels=data/tokenize/${short}-ud-train.toklabels
    label_type=label_file
    eval_file="--txt_file data/tokenize/${short}-ud-${mode}.txt"
fi

if [[ "$args" == *"--save_dir"* ]]; then
    savedir=""
else
    savedir="--save_dir ${outputprefix}saved_models/tokenize"
fi

# copy initial text data
if [ ! -e data/tokenize/${short}-ud-${mode}.txt ]; then
    echo 'copying test data for: '${treebank}
    cp ${UDBASE}/${treebank}/${short}-ud-${mode}.txt data/tokenize
fi

ete_file=data/ete/${short}.${mode}.pred.ete.conllu
# run the tokenizer
echo 'running tokenizer...'
CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.tokenizer --mode predict $eval_file --lang $lang --conll_file data/tokenize/${short}.${mode}.${outputprefix}pred.ete.conllu --shorthand $short $savedir $args
cp data/tokenize/${short}.${mode}.${outputprefix}pred.ete.conllu $ete_file

# run the mwt expander
if [ -e saved_models/mwt/${short}_mwt_expander.pt ]; then
    echo 'running mwt expander...'
    CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.mwt_expander --mode predict --eval_file $ete_file --shorthand $short --output_file data/mwt/${short}.${mode}.pred.ete.conllu --save_dir saved_models/mwt $args
    cp data/mwt/${short}.${mode}.pred.ete.conllu $ete_file
fi

# run the part-of-speech tagger
echo 'running part-of-speech tagger...'
CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.tagger --eval_file $ete_file --output_file data/pos/${short}.${mode}.${outputprefix}pred.ete.conllu --lang $short --shorthand $short --mode predict --save_dir saved_models/pos
cp data/pos/${short}.${mode}.${outputprefix}pred.ete.conllu $ete_file

# run the lemmatizer
echo 'running lemmatizer...'
CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.lemmatizer --data_dir data/lemma --eval_file $ete_file --output_file data/lemma/${short}.${mode}.${outputprefix}pred.ete.conllu --lang $short --mode predict 
cp data/lemma/${short}.${mode}.${outputprefix}pred.ete.conllu $ete_file 

# run the dependency parser
echo 'running dependency parser...'
CUDA_VISIBLE_DEVICES=$gpu python -m stanfordnlp.models.parser --eval_file $ete_file --output_file data/depparse/${short}.${mode}.${outputprefix}pred.ete.conllu --lang $short --shorthand $short --mode predict --save_dir saved_models/depparse
cp data/depparse/${short}.${mode}.${outputprefix}pred.ete.conllu $ete_file

# get final output table
# copy over gold file
cp ${UDBASE}/${treebank}/${short}-ud-${mode}.conllu data/ete
gold_file=data/ete/${short}-ud-${mode}.conllu
# run official eval script
echo 'running official eval script'
python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $ete_file
