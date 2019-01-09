#!/bin/bash

inputfile=$1; shift
outputfile=$1; shift
treebank=$1; shift
module=$1; shift

UDPIPEDIR=/u/nlp/data/dependency_treebanks/CoNLL18/udpipe-1.2.0-bin
SHORT2TB=./short_to_tb

python -m models.udpipe_wrapper --input_file $inputfile --output_file $outputfile --treebank $treebank --module $module --udpipe_dir $UDPIPEDIR --short2tb $SHORT2TB

