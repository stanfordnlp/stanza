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
    $PYTHON scripts/sentiment/parse_arguana_xml.py extern_data/sentiment/arguana/arguana-tripadvisor-annotated-v2/split/training extern_data/sentiment/arguana/train.txt

    echo "MELD"
    $PYTHON scripts/sentiment/convert_MELD.py extern_data/sentiment/MELD/MELD/train_sent_emo.csv extern_data/sentiment/MELD/train.txt
    $PYTHON scripts/sentiment/convert_MELD.py extern_data/sentiment/MELD/MELD/dev_sent_emo.csv extern_data/sentiment/MELD/dev.txt
    $PYTHON scripts/sentiment/convert_MELD.py extern_data/sentiment/MELD/MELD/test_sent_emo.csv extern_data/sentiment/MELD/test.txt

    echo "SLSD"
    $PYTHON scripts/sentiment/convert_slsd.py extern_data/sentiment/slsd/slsd extern_data/sentiment/slsd/train.txt

    echo "airline"
    $PYTHON scripts/sentiment/convert_airline.py extern_data/sentiment/airline/Tweets.csv extern_data/sentiment/airline/train.txt
    # TODO: add the corenlp converstion script here
elif [ "$language" = "german" ]; then
    echo "PROCESSING GERMAN"
    echo "Scare"
    python3 -m scripts.sentiment.process_scare extern_data/sentiment/german/scare
else
    echo "Unknown language $language"
fi
