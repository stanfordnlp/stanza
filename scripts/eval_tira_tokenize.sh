lang=$1
rawfile=$2
outputfile=$3
short=$4
shift;shift;shift;shift;
args=$@

if [[ `hostname` = *"stanford.edu" ]]; then
    SAVE_DIR=final_saved_models/tokenize
    DATA_DIR=final_data/tokenize
    #SAVE_DIR=final_saved_models2/tokenize
    #DATA_DIR=data/tokenize
    PYTHON=python
    CUDA=""
else
    SAVE_DIR=/media/data/final_saved_models/tokenize
    DATA_DIR=/media/data/final_data/tokenize
    PYTHON=$HOME/anaconda3/bin/python
    CUDA="--no_cuda"
fi

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd $ROOT

if [ $lang == "vi" ]; then
    # first-pass syllable segmentation for Vietnamese
    $PYTHON utils/postprocess_vietnamese_tokenizer_data.py $rawfile -o $outputfile
    eval_file="--json_file $outputfile"
else
    eval_file="--txt_file $rawfile"
fi

$PYTHON -m models.tokenizer --mode predict $eval_file --lang $lang --conll_file $outputfile --vocab_file ${DATA_DIR}/${short}_vocab.pkl --shorthand $short $CUDA --save_dir ${SAVE_DIR} $args
