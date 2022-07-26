---
layout: page
title: Sentiment Analysis
keywords: sentiment, classifier
permalink: '/sentiment.html'
nav_order: 11
parent: Neural Pipeline
---

## Description

Sentiment is added to the stanza pipeline by using [a CNN classifier](https://arxiv.org/abs/1408.5882).

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| sentiment | SentimentProcessor | tokenize | `sentiment` | Adds the `sentiment` annotation to each [`Sentence`](data_objects.md#sentence) in the `Document` |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| 'model_path' | string | depends on the language | Where to load the model. |
| 'pretrain_path' | string | depends on the language | Which set of pretrained word vectors to use. Can be changed for existing models, but this is not recommended, as the models are trained to work specifically with one set of word vectors. |
| 'batch_size' | int | None | If None, run everything at once.  If set to an integer, break processing into chunks of this size |

## Example Usage

The `SentimentProcessor` adds a label for sentiment to each
[`Sentence`](data_objects.md#sentence).  The existing models each
support negative, neutral, and positive, represented by 0, 1, 2
respectively.  Custom models could support any set of labels as long
as you have training data.

### Simple code example


```python
import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
doc = nlp('I hate that they banned Mox Opal')
for i, sentence in enumerate(doc.sentences):
    print("%d -> %d" % (i, sentence.sentiment))
```

The output produced (aside from logging) will be:

```
0 -> 0
```

This represents a negative sentiment.

In some cases, such as datasets with one sentence per line or twitter
data, you want to guarantee that there is one sentence per document
processed.  You can do this [by turning off the sentence
splitting](tokenize.html#tokenization-without-sentence-segmentation).

```python
import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment', tokenize_no_ssplit=True)
doc = nlp('Jennifer has pretty antennae.  I hope I meet her someday')
for i, sentence in enumerate(doc.sentences):
    print("%d -> %d" % (i, sentence.sentiment))
```

The output produced (aside from logging) will be:

```
0 -> 2
```

This represents a positive sentiment.

## Available models

There are currently three models available: English, Chinese, and German.

### English

English is trained on the following data sources:

[Stanford Sentiment Treebank](https://github.com/stanfordnlp/sentiment-treebank), including extra training sentences

[MELD](https://github.com/declare-lab/MELD/tree/master/data/MELD), text only

[SLSD](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)

[Arguana](http://argumentation.bplaced.net/arguana/data)

[Airline Twitter Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment/data)

The score on this model is not directly comparable to existing SST
models, as this is using a 3 class projection of the 5 class data and
includes several additional data sources (hence the `sstplus`
designation).  However, training this model on 2 class data using
higher dimension word vectors achieves the 87 score reported in the
original CNN classifier paper.  On a three class projection of the
SST test data, the model trained on multiple datasets gets 70.0%.

### Chinese

The Chinese model is trained using the polarity signal from the following 

http://a1-www.is.tokushima-u.ac.jp/member/ren/Ren-CECps1.0/Ren-CECps1.0.html

We were unable to find standard scores or even standard splits for this dataset.

Using the gsdsimp word vectors package, training with extra trained
word vectors added to the existing word vectors, we built a model
which gets 0.694 test accuracy on a random split of the training data.
The split can be recreated using process_ren_chinese.py.

### German

The German model is build from
[sb10k](https://www.spinningbytes.com/resources/germansentiment/),
a dataset of German tweets.

The original sb10k paper cited an F1 score of 65.09.  Without using
the distant learning step, but using the SB word vectors, we acheived
63 F1.  The included model uses the standard German word2vec vectors
and only gets 60.5 F1.  We considered this acceptable instead of
redistributing the much larger tweet word vectors.

We tried training with the longer snippets of text from
[Usage](https://www.romanklinger.de/usagecorpus/) and
[Scare](https://www.romanklinger.de/scare/), but this seemed to have a
noticeable negative effect on the accuracy.
