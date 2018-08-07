lang=$1
inputfile=$2
outputfile=$3
short=$4
shift;shift;shift;shift;

if [[ `hostname` = *"stanford.edu" ]]; then
    SAVE_DIR=/u/scr/tdozat/v3saves/CoNLL18
    PYTHON=python
    ROOT=/u/scr/pengqi/Parser-v3
    SHORT_TO_TB=/u/scr/pengqi/UD_from_scratch/short_to_tb
else
    SAVE_DIR=/media/data/final_saved_models/depparse
    PYTHON=$HOME/anaconda3/bin/python
    ROOT=$HOME/Parser-v3
    SHORT_TO_TB=/media/data/short_to_tb
fi

if [[ "$inputfile" == *"ja_modern"* ]]; then
    ADDITIONAL_FLAGS="--CoNLLUDataset batch_size=5000"
else
    ADDITIONAL_FLAGS="--CoNLLUDataset batch_size=1000 --SubtokenVocab max_buckets=3"
fi

ROOT0="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd $ROOT

fullname=`grep -e "^$short " $SHORT_TO_TB | awk '{print $2}'`

$PYTHON main.py --save_dir $SAVE_DIR/$fullname/Parser --save_metadir $SAVE_DIR run $inputfile $ADDITIONAL_FLAGS | $PYTHON ${ROOT0}/utils/post_insert_mwt.py $inputfile $outputfile
