#!/bin/bash
#
# Download word vector files for all supported languages. Run as:
#   ./download_vectors.sh WORDVEC_DIR
# where WORDVEC_DIR is the target directory to store the word vector data.

# check arguments
: ${1?"Usage: $0 WORDVEC_DIR"}
WORDVEC_DIR=$1

# constants and functions
CONLL17_URL="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989/word-embeddings-conll17.tar"
CONLL17_TAR="word-embeddings-conll17.tar"

FASTTEXT_BASE_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-wiki"

# TODO: some fasttext vectors are now at
# https://fasttext.cc/docs/en/pretrained-vectors.html
# there are also vectors for
# Welsh, Icelandic, Thai, Sanskrit
# https://fasttext.cc/docs/en/crawl-vectors.html

# We get the Armenian word vectors from here:
# https://github.com/ispras-texterra/word-embeddings-eval-hy
# https://arxiv.org/ftp/arxiv/papers/1906/1906.03134.pdf
# In particular, the glove model (dogfooding):
# https://at.ispras.ru/owncloud/index.php/s/pUUiS1l1jGKNax3/download
# These vectors improved F1 by about 1 on various tasks for Armenian
# and had much better coverage of Western Armenian

declare -a FASTTEXT_LANG=("Afrikaans" "Breton" "Buryat" "Chinese" "Faroese" "Gothic" "Kurmanji" "North_Sami" "Serbian" "Upper_Sorbian")
declare -a FASTTEXT_CODE=("af" "br" "bxr" "zh" "fo" "got" "ku" "se" "sr" "hsb")
declare -a LOCAL_CODE=("af" "br" "bxr" "zh" "fo" "got" "kmr" "sme" "sr" "hsb")

color_green='\033[32;1m'
color_clear='\033[0m' # No Color
function msg() {
    echo -e "${color_green}$@${color_clear}"
}

function prepare_fasttext_vec() {
    lang=$1
    ftcode=$2
    code=$3

    cwd=$(pwd)
    mkdir -p $lang
    cd $lang
    msg "=== Downloading fasttext vector file for ${lang}..."
    url="${FASTTEXT_BASE_URL}/wiki.${ftcode}.vec"
    fname="${code}.vectors"
    wget $url -O $fname

    msg "=== Compressing file ${fname}..."
    xz $fname
    cd $cwd
}

# do the actual work
mkdir -p $WORDVEC_DIR
cd $WORDVEC_DIR

msg "Downloading CONLL17 word vectors. This may take a while..."
wget $CONLL17_URL -O $CONLL17_TAR

msg "Extracting CONLL17 word vector files..."
tar -xvf $CONLL17_TAR
rm $CONLL17_TAR

msg "Preparing fasttext vectors for the rest of the languages."
for (( i=0; i<${#FASTTEXT_LANG[*]}; ++i)); do
    prepare_fasttext_vec ${FASTTEXT_LANG[$i]} ${FASTTEXT_CODE[$i]} ${LOCAL_CODE[$i]}
done

# handle old french
mkdir Old_French
ln -s French/fr.vectors.xz Old_French/fro.vectors.xz

msg "All done."
