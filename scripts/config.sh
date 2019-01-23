#!/bin/bash
#
# Set environment variables for the training and testing of stanfordnlp modules.

# Set UDBASE to the location of CoNLL18 folder
# For details, see http://universaldependencies.org/conll18/data.html
# export UDBASE=/path/to/CoNLL18

# Set directories to store processed training/evaluation files
export DATA_ROOT=./data
export TOKENIZE_DATA_DIR=$DATA_ROOT/tokenize
export MWT_DATA_DIR=$DATA_ROOT/mwt
export LEMMA_DATA_DIR=$DATA_ROOT/lemma
export POS_DATA_DIR=$DATA_ROOT/pos
export DEPPARSE_DATA_DIR=$DATA_ROOT/depparse
export ETE_DATA_DIR=$DATA_ROOT/ete

# Link to external resources
#if [ ! -e extern_data/CoNLL18 ]; then
    #ln -s /path/to/CoNLL18 extern_data/CoNLL18
#fi

#if [ ! -e extern_data/word2vec ]; then
    #ln -s /path/to/word2vec extern_data/word2vec
#fi
