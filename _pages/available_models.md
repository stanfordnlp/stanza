---
layout: page
title: Available Models & Languages
keywords: available models
permalink: '/available_models.html'
nav_order: 1
parent: Models
---

Stanza provides pretrained NLP models for a total of 80 human languages. On this page we provide detailed information on these models.

Pretrained models in Stanza can be divided into two categories, based on the datasets they were trained on:
1. Universal Dependencies (UD) models, which are trained on the UD treebanks, and cover functionalities including tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging and dependency parsing;
2. NER models, which support named entity tagging for 8 languages, and are trained on various NER datasets.

## Available UD Models

Tokenization, MWT (if applicable), POS, Lemma, and dependency parsing
is provided using data from
[Universal Dependencies v2.12](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5150).
Results for these models are on the
[performance](https://stanfordnlp.github.io/stanza/performance.html) page.
There are also past results from previous versions of the models.

## Combined Models

For some languages, [we built models out of multiple datasets at once](combined_models.md), getting wider coverage and better performance.  New languages can be added on request.  When available, the combined models are the defaults.

## Other Available Pipelines

### Myanmar

The [Asian Language Treebank Project](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/) has a Myanmar dataset.
We used this dataset to build a tokenizer for Myanmar.

Stanza includes a script which converts the trees and the officially proposed train/dev/split to an ersatz UD dataset:

https://github.com/stanfordnlp/stanza/blob/v1.5.1/stanza/utils/datasets/tokenization/convert_my_alt.py

### Sindhi

The NLP team at [ISRA](https://isra.edu.pk/) graciously provided us with several passages of tokenized Sindhi text.
We used this to add a tokenizer for Sindhi to Stanza.  This is particularly useful in that it
allows us to incorporate a Sindhi NER model.

With permission, we are currently [hosting the Sindhi tokenization data on StanfordNLP's github](https://github.com/stanfordnlp/sindhi-tokenization).

### Thai

We have trained a couple Thai tokenizer models based on publicly
available datasets.  The Inter-BEST dataset had some strange sentence
tokenization according to the authors of pythainlp, so we used their
software to resegment the sentences before training.  As this is a
questionable standard to use, we made the Orchid tokenizer the
default.

| Dataset | Token Accuracy | Sentence Accuracy |  Notes |
| :------ | :------------- | :---------------- | :----- |
| Orchid  | 87.98          |  70.99            |        |
| BEST    | 95.73          |  77.93            | Sentences are re-split using pythainlp |

New in v1.10.1
{: .label .label-green }

The newly created [TUD treebank](https://github.com/nlp-chula/TUD)
has UPOS and dependencies, and of course can be used for training a
tokenizer.  The tokenizer built from this has a *terrible* sentence
split rate, somewhere around 20%.  The POS and dependencies are
reasonable, though:

| Embedding | dev UPOS  | dev depparse LAS |
| :------ | :---------: | :--------------: |
| no transformer |      91.26    | 73.57 |
| [wangchanberta](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)  |      92.21    | 76.65 |


> Panyut Sriwirote, Wei Qi Leong, Charin Polpanumas, Santhawat Thanyawong, William Chandra Tjhi, Wirote Aroonmanakun, and Attapol T. Rutherford.  [The Thai Universal Dependency Treebank](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00745/128939/The-Thai-Universal-Dependency-Treebank).  In Transactions of the Association for Computational Linguistics.  2025. \[[pdf](https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl%5C_a%5C_00745/2514583/tacl%5C_a%5C_00745.pdf)\]
{: .citation }

## Available NER Models

A description of the models available for the NER tool, along with their performance on test datasets, can be found [here](ner_models.md).

## Available Sentiment Models

A description of the sentiment tool and the models available for that tool can be found [here](sentiment.md).

## Available Conparse Models

A description of the constituency parser and the models available for that tool can be found [here](constituency.md).

## Training New Models

To train new models, please see the documents on [training](training.md) and [adding a new language](new_language.md).
