lang=$1
rawfile=$2
outputfile=$3
treebank=$4
shift;shift;shift;shift;
args=$@

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd $ROOT

if [ $lang == "vi" ]; then
    # first-pass syllable segmentation for Vietnamese
    python utils/postprocess_vietnamese_tokenizer_data.py $rawfile -o $outputfile
    eval_file="--json_file $outputfile"
else
    eval_file="--txt_file $rawfile"
fi

python -m models.tokenizer --mode predict $eval_file --lang $lang --mwt_json_file data/tokenize/${short}-ud-train-mwt.json --conll_file $outputfile --vocab_file data/tokenize/${short}_vocab.pkl --save_name $SAVE_NAME --shorthand $short --no-cuda $args
