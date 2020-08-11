
# Run classifier.py to build a sentiment model
# Options:
#   -b <base directory>
#   -d sst_fiveclass | sst_binary | sst_threeclass
#   -e : eval mode
#   -t : train mode

SST_BINARY_DIR=$SENTIMENT_DATA_DIR/sst-processed/binary
SST_THREECLASS_DIR=$SENTIMENT_DATA_DIR/sst-processed/threeclass
SST_FIVECLASS_DIR=$SENTIMENT_DATA_DIR/sst-processed/fiveclass

basedir=$SST_FIVECLASS_DIR
mode="train"

while getopts "b:d:et" OPTION
do
  case $OPTION in
  b)
    basedir=$OPTARG
    ;;
  d)
    if [ "$OPTARG" = "sst_fiveclass" ]; then
      basedir=$SST_FIVECLASS_DIR
    elif [ "$OPTARG" = "sst_binary" ]; then
      basedir=$SST_BINARY_DIR
    elif [ "$OPTARG" = "sst_threeclass" ]; then
      basedir=$SST_THREECLASS_DIR
    else
      echo "Unknown dataset $OPTARG"
      exit  
    fi
    ;;
  e)
    mode="eval"
    ;;
  t)
    mode="train"
    ;;
  esac
done

shift $((OPTIND -1))
args=$@

if hash python3 2>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

if [ "$mode" = "train" ]; then
  $PYTHON -u -m stanza.models.classifier --train_file $basedir/train.txt --dev_file $basedir/dev.txt --test_file $basedir/test.txt $args
else
  $PYTHON -u -m stanza.models.classifier --no_train --test_file $basedir/test.txt $args
fi
