lang=$1
inputfile=$2
outputfile=$3
short=$4
shift;shift;shift;shift;

if [[ `hostname` = *"stanford.edu" ]]; then
    SAVE_DIR=/u/scr/tdozat/v3saves
    PYTHON=python
    ROOT=~/scr/Parser-v3
else
    SAVE_DIR=/media/data/final_saved_models/depparse
    PYTHON=$HOME/anaconda3/bin/python
    ROOT=$HOME/Parser-v3
fi

cd $ROOT

fullname=`grep -e "^$short " /media/data/short_to_tb | awk '{print $2}'`

$PYTHON main.py --save_dir $SAVE_DIR/$fullname/Tagger run --output_dir `dirname $outputfile` --output_filename `basename $outputfile` TaggerNetwork $inputfile
