treebank=$1
shift
eval_treebank=$1
shift
gpu=$1
shift
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
eval_short=`bash scripts/treebank_to_shorthand.sh ud $eval_treebank`
lang=`echo $short | sed -e 's#_.*##g'`
args=$@
if [ $lang == "vi" ]; then
    labels=data/tokenize/${short}-ud-train.json
    label_type=json_file
    eval_file="--json_file data/tokenize/${short_eval}-ud-dev.json"
else
    labels=data/tokenize/${short}-ud-train.toklabels
    label_type=label_file
    eval_file="--txt_file /u/nlp/data/dependency_treebanks/CoNLL18/$eval_treebank/${eval_short}-ud-dev.txt"
fi

if [ ! -e $labels ]; then
    bash scripts/prep_tokenize_data.sh $treebank train
    if [ $lang == "vi" ]; then
        bash scripts/prep_tokenize_data.sh $treebank dev
    fi
fi

SAVE_DIR=saved_models/tokenize
SAVE_NAME=${short}.tok.pkl
DEV_GOLD=/u/nlp/data/dependency_treebanks/CoNLL18/${eval_treebank}/${eval_short}-ud-dev.conllu
seqlen=$(python -c "from math import ceil; print(ceil($(python utils/avg_sent_len.py $labels) * 3 / 100) * 100)")
echo "Evaluating $args..."
CUDA_VISIBLE_DEVICES=$gpu python -m models.tokenizer --mode predict $eval_file --lang $lang --mwt_json_file data/tokenize/${short}-ud-train-mwt.json --conll_file data/tokenize/${eval_short}.dev.pred.conllu --vocab_file data/tokenize/${short}_vocab.pkl --save_dir $SAVE_DIR --save_name $SAVE_NAME --shorthand $short $args
results=`python utils/conll18_ud_eval.py -v $DEV_GOLD data/tokenize/${eval_short}.dev.pred.conllu | head -5 | tail -n+3 | awk '{print $7}' | pr --columns 3 -aJT`
echo $results $args >> data/tokenize/${eval_short}.results
echo $eval_short $results $args
