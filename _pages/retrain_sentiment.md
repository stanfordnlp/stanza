---
layout: page
title: Retrain models for a Sentiment dataset
keywords: stanza, sentiment model training
permalink: '/retrain_sentiment.html'
nav_order: 12
parent: Training
---

## End to End Retraining Example

As with NER, it is easier to retrain an existing model, as there should be no code changes needed to rebuild the model.

For this example, we will recreate the base English sentiment model,
trained on Stanford Sentiment Treebank, without the additional pieces
that make up the "sstplus" model we distribute.

### Environment

First, please
[set up your environment](https://stanfordnlp.github.io/stanza/new_language_sentiment.html#environment)
in the same manner as for building a completely new sentiment model.

### Download data

For most of the datasets, you will need to manually download the data.
Instructions for each dataset Stanza knows how to process are at the
top of the `prepare_sentiment_dataset.py` script:

https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/datasets/sentiment/prepare_sentiment_dataset.py

For example, there are specific (and very simple) instructions on
where to download the SST dataset.  We have a version which is close
to the original while fixing several tokenization errors and
re-treebanking some broken trees:

```
cd $SENTIMENT_BASE
git clone git@github.com:stanfordnlp/sentiment-treebank.git
```

### Prepare data

The data preparation script for SST requires
[installing CoreNLP](https://stanfordnlp.github.io/stanza/client_setup.html),
as it was written at a time when there was no constituency parser in
Stanza and therefore no support for constituency trees such as SST.
PRs to replace that are welcome!

Once CoreNLP is installed, a three class version of the dataset can be converted with:

```
python3 stanza/utils/datasets/sentiment/prepare_sentiment_dataset.py en_sst3
```

This converts the trees to a json format usable by the classifier code.

### Train model

Once the data is prepared, the model can be trained using a command line such as:

```
python3 stanza/utils/training/run_sentiment.py en_sst3
```

By default, this will not use any transformer.  A default transformer can be used with the `--use_bert` flag, such as

```
python3 stanza/utils/training/run_sentiment.py en_sst3 --use_bert
```

If you want to use different word vectors, the flag for that is

```
python3 stanza/utils/training/run_sentiment.py en_sst3 --wordvec_pretrain_file <filename>
```

### Test model

To test the model, you can use the `--score_dev` or `--score_test` flags as appropriate.

