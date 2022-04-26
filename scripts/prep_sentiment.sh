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
    echo "--- ArguAna ---"
    $PYTHON -m stanza.utils.datasets.sentiment.process_arguana_xml extern_data/sentiment/arguana/arguana-tripadvisor-annotated-v2/split/training $SENTIMENT_DATA_DIR en_arguana

    echo "--- MELD ---"
    $PYTHON -m stanza.utils.datasets.sentiment.process_MELD extern_data/sentiment/MELD $SENTIMENT_DATA_DIR en_meld

    echo "--- SLSD ---"
    $PYTHON -m stanza.utils.datasets.sentiment.process_slsd extern_data/sentiment/slsd $SENTIMENT_DATA_DIR en_slsd

    echo "--- AIRLINE ---"
    $PYTHON -m stanza.utils.datasets.sentiment.process_airline extern_data/sentiment/airline $SENTIMENT_DATA_DIR en_airline

    echo "--- SST ---"
    $PYTHON -m stanza.utils.datasets.sentiment.process_sst
elif [ "$language" = "german" ]; then
    echo "PROCESSING GERMAN"
    echo "Scare"
    $PYTHON -m stanza.utils.datasets.sentiment.process_scare extern_data/sentiment/german/scare $SENTIMENT_DATA_DIR de_scare
    echo "Usage"
    $PYTHON -m stanza.utils.datasets.sentiment.process_usage_german extern_data/sentiment/USAGE $SENTIMENT_DATA_DIR de_usage
    echo "SB-10k"
    $PYTHON -m stanza.utils.datasets.sentiment.process_sb10k --csv_filename extern_data/sentiment/german/sb-10k/de_full/de_test.tsv --out_dir $SENTIMENT_DATA_DIR --short_name de_sb10k --split test --sentiment_column 2 --text_column 3
    $PYTHON -m stanza.utils.datasets.sentiment.process_sb10k --csv_filename extern_data/sentiment/german/sb-10k/de_full/de_train.tsv --out_dir $SENTIMENT_DATA_DIR --short_name de_sb10k --split train_dev --sentiment_column 2 --text_column 3
    #$PYTHON -m scripts.sentiment.process_sb10k --csv_filename extern_data/sentiment/german/sb-10k/sb_10k.tsv --out_dir extern_data/sentiment/german/sb-10k
elif [ "$language" = "chinese" ]; then
    echo "PROCESSING CHINESE"
    echo "Ren-CECps"
    $PYTHON -m stanza.utils.datasets.sentiment.process_ren_chinese extern_data/sentiment/chinese/RenCECps $SENTIMENT_DATA_DIR zh_ren
else
    echo "Unknown language $language"
fi
