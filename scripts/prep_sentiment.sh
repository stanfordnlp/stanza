if hash python3 2>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

if [ -z "$SENTIMENT_DATA_DIR" ]; then
    source scripts/config.sh
fi

# Manipulates various downloads from their original form to a form
# usable by the classifier model

# Notes on the individual datasets can be found in the relevant
# process_dataset script

# Run as follows:
# prep_sentiment -lenglish
# prep_sentiment -lgerman

language="english"
while getopts "l:" OPTION
do
    case $OPTION in
    l)
        language=$OPTARG
    esac
done

if [ "$language" = "english" ]; then
    echo "PROCESSING ENGLISH"
    echo "ArguAna"
    $PYTHON scripts/sentiment/process_arguana_xml.py $SENTIMENT_DATA_DIR/arguana/arguana-tripadvisor-annotated-v2/split/training $SENTIMENT_DATA_DIR/arguana/train.txt

    echo "MELD"
    $PYTHON scripts/sentiment/process_MELD.py $SENTIMENT_DATA_DIR/MELD/MELD/train_sent_emo.csv $SENTIMENT_DATA_DIR/MELD/train.txt
    $PYTHON scripts/sentiment/process_MELD.py $SENTIMENT_DATA_DIR/MELD/MELD/dev_sent_emo.csv $SENTIMENT_DATA_DIR/MELD/dev.txt
    $PYTHON scripts/sentiment/process_MELD.py $SENTIMENT_DATA_DIR/MELD/MELD/test_sent_emo.csv $SENTIMENT_DATA_DIR/MELD/test.txt

    echo "SLSD"
    $PYTHON scripts/sentiment/process_slsd.py $SENTIMENT_DATA_DIR/slsd/slsd $SENTIMENT_DATA_DIR/slsd/train.txt

    echo "airline"
    $PYTHON -m scripts.sentiment.process_airline $SENTIMENT_DATA_DIR/airline/Tweets.csv $SENTIMENT_DATA_DIR/airline/train.txt

    echo "sst"
    if [ -z "$SENTIMENT_SST_HOME" ]; then
        SENTIMENT_SST_HOME=$SENTIMENT_DATA_DIR/sentiment-treebank
        echo "  Assuming git download of SST is at " $SENTIMENT_SST_HOME
    fi
    scripts/sentiment/process_sst.sh -i$SENTIMENT_SST_HOME -o$SENTIMENT_DATA_DIR/sst-processed
elif [ "$language" = "german" ]; then
    echo "PROCESSING GERMAN"
    echo "Scare"
    $PYTHON -m scripts.sentiment.process_scare $SENTIMENT_DATA_DIR/german/scare
    echo "Usage"
    $PYTHON -m scripts.sentiment.process_usage_german extern_data/sentiment/USAGE
    echo "SB-10k"
    $PYTHON  -m scripts.sentiment.process_sb10k --csv_filename extern_data/sentiment/german/sb-10k/de_full/de_test.tsv --out_dir extern_data/sentiment/german/sb-10k --split test --sentiment_column 2 --text_column 3
    $PYTHON  -m scripts.sentiment.process_sb10k --csv_filename extern_data/sentiment/german/sb-10k/de_full/de_train.tsv --out_dir extern_data/sentiment/german/sb-10k --split train_dev --sentiment_column 2 --text_column 3
    #$PYTHON -m scripts.sentiment.process_sb10k --csv_filename extern_data/sentiment/german/sb-10k/sb_10k.tsv --out_dir extern_data/sentiment/german/sb-10k
elif [ "$language" = "chinese" ]; then
    echo "PROCESSING CHINESE"
    echo "Ren-CECps"
    $PYTHON -m scripts.sentiment.process_ren_chinese extern_data/sentiment/chinese/RenCECps extern_data/sentiment/chinese/RenCECps
else
    echo "Unknown language $language"
fi
