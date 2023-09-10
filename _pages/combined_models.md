---
layout: page
title: Combined models
keywords: combined models
permalink: '/combined_models.html'
nav_order: 5
parent: Models
---

## Combined models

The default models for some languages are "combined".  The goal is to get better coverage of the language, hopefully without sacrificing consistency in the annotation scheme.  In each case, the data used to train the models is a combination of multiple UD datasets.

| Language | Datasets | Other |
| :------- | :------------ | :------ |
| English | EWT, GUM, GUMReddit, PUD, Pronouns | |
| French  | GSD, ParisStories, Rhapsodie, Sequoia | |
| Hebrew  | IAHLTwiki | [HTB fork from IAHLT](https://github.com/IAHLT/UD_Hebrew) |
| Italian | ISDT, VIT, PoSTWITA, and TWITTIRO | [MWT list from Prof. Attardi](https://github.com/stanfordnlp/handparsed-treebank/blob/master/italian-mwt/italian.mwt) |

Other data sets would be added, or combined models for other languages
created, but there are often data problems preventing that.  For
example, English Lines uses a different xpos and feature scheme.
Spanish GSD and AnCora use the same general annotation scheme, but
sentence splits are annotated differently, and some of the features
are noticeably different.  Hopefully over time we can resolve some of
those issues and expand the models.

Whether or not this was a good idea was explored [in a GURT paper from Georgetown](https://arxiv.org/abs/2302.00636)

## Data augmentation

In general, the models also use various sorts of text "data
augmentation" of the original data.

For example, in a language where all of the sentences in the POS
training data end with punctuation, we remove the trailing punctuation
from some fragment of sentences to train the model to handle sentences
which don't end with sentence final punctuation.  Otherwise, we would
frequently get issues such as a model tagging `This is an unfinished
sentence_PU`

The tokenization models also use a couple different versions of this.
For example, most of the tokenization datasets have just one or a
couple forms of quotes, but we replace some fraction of quotes with
different types so that each model has a chance of correctly
tokenizing `«data augmentation»`, `"data augmentation"`, etc.  Also,
sentence final punctuation will often have spaces added or removed to
make the model more robust to typos.
