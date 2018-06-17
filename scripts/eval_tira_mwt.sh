lang=$1
inputfile=$2
outputfile=$3
short=$4
shift;shift;shift;shift;
args=$@

if [[ `hostname` = *"stanford.edu" ]]; then
    SAVE_DIR=final_saved_models/mwt
    DATA_DIR=data/mwt
    PYTHON=python
else
    SAVE_DIR=/media/data/final_saved_models/mwt
    DATA_DIR=/media/data/final_data/mwt
    PYTHON=$HOME/anaconda3/bin/python
fi

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd $ROOT

echo "Evaluating $args..."
$PYTHON -m models.mwt_expander --eval_file $inputfile --data_dir ${DATA_DIR} \
    --output_file $outputfile --lang $lang --shorthand $short --mode predict --save_dir $SAVE_DIR --cpu $args
