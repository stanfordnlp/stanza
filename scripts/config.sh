#!/bin/bash
#
# Set environment variables for the training and testing of stanza modules.

# Set UDBASE to the location of UD data folder
# The data should be CoNLL-U format
# For details, see
#   http://universaldependencies.org/conll18/data.html (CoNLL-18 UD data)
#   https://universaldependencies.org/
# When rebuilding models based on Universal Dependencies, download the
#   UD data to some directory, set UDBASE to that directory, and
#   uncomment this line.  Alternatively, put UDBASE in your shell
#   config, Windows env variables, etc as relevant.
# export UDBASE=/path/to/UD

# Set NERBASE to the location of NER data folder
# The data should be BIO format or convertable to that format
# For details, see https://www.aclweb.org/anthology/W03-0419.pdf (CoNLL-03 NER paper)
# There are other NER datasets, supported in
#   stanza/utils/datasets/ner/prepare_ner_dataset.py
# If rebuilding NER data, choose a location for the NER directory
#   and set NERBASE to that variable.
# export NERBASE=/path/to/NER

# Set CONSTITUENCY_BASE to the location of NER data folder
# The data will be in some dataset-specific format
# There is a conversion script which will turn this
#   into a PTB style format
#   stanza/utils/datasets/constituency/prepare_con_dataset.py
# If processing constituency data, choose a location for the CON data
#   and set CONSTITUENCY_BASE to that variable.
# export CONSTITUENCY_BASE=/path/to/CON

# Set directories to store processed training/evaluation files
# $DATA_ROOT is a default home for where all the outputs from the
#   preparation scripts will go.  The training scripts will then look
#   for the stanza formatted data in that directory.
export DATA_ROOT=./data
export TOKENIZE_DATA_DIR=$DATA_ROOT/tokenize
export MWT_DATA_DIR=$DATA_ROOT/mwt
export LEMMA_DATA_DIR=$DATA_ROOT/lemma
export POS_DATA_DIR=$DATA_ROOT/pos
export DEPPARSE_DATA_DIR=$DATA_ROOT/depparse
export ETE_DATA_DIR=$DATA_ROOT/ete
export NER_DATA_DIR=$DATA_ROOT/ner
export CHARLM_DATA_DIR=$DATA_ROOT/charlm
export CONSTITUENCY_DATA_DIR=$DATA_ROOT/constituency
export SENTIMENT_DATA_DIR=$DATA_ROOT/sentiment

# Set directories to store external word vector data
export WORDVEC_DIR=./extern_data/wordvec
