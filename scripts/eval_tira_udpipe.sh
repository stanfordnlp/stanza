lang=$1
inputfile=$2
outputfile=$3
short=$4
module=$5
shift;shift;shift;shift;shift
args=$@

if [[ `hostname` = *"stanford.edu" ]]; then
    UDPIPE_DIR=/u/nlp/data/dependency_treebanks/CoNLL18/udpipe-1.2.0-bin
    SHORT2TB=./short_to_tb
    PYTHON=python
else
    UDPIPE_DIR=/media/data/udpipe-1.2.0-bin
    SHORT2TB=/media/data/short_to_tb
    PYTHON=$HOME/anaconda3/bin/python
fi

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd $ROOT

echo "Starting $module with UDPipe..."
$PYTHON -m models.udpipe_wrapper --input_file $inputfile --output_file $outputfile --treebank $short --module $module --udpipe_dir $UDPIPE_DIR --short2tb $SHORT2TB

