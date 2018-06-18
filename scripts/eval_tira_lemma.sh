lang=$1
inputfile=$2
outputfile=$3
short=$4
shift;shift;shift;shift;
args=$@

if [[ `hostname` = *"stanford.edu" ]]; then
    SAVE_DIR=final_saved_models/lemma
    DATA_DIR=data/lemma
    PYTHON=python
else
    SAVE_DIR=/media/data/final_saved_models/lemma
    DATA_DIR=/media/data/final_data/lemma
    PYTHON=$HOME/anaconda3/bin/python
fi

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd $ROOT

echo "Evaluating $args..."
if [[ "$lang" == "vi" || "$lang" == "fro" ]]; then
    python -m models.identity_lemmatizer --data_dir $DATA_DIR --eval_file $inputfile --mode predict \
        --output_file $outputfile --lang $short --model_dir $SAVE_DIR
else
    python -m models.lemmatizer --data_dir $DATA_DIR --eval_file $inputfile \
        --output_file $outputfile --lang $short --mode predict --model_dir $SAVE_DIR --cpu
fi
#python utils/conll18_ud_eval.py -v $gold_file $DATADIR/$output_file | grep "Lemmas" | awk '{print $7}'
