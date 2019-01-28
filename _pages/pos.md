---
title: POSProcessor 
keywords: pos
permalink: '/pos.html'
---

## Description

Labels tokens with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html).

| Property name | Annotator class name | Generated Annotation |
| --- | --- | --- |
| pos | POSProcessor | UPOS, XPOS, and UFeats annotations accessible through [`Word`](data_objects.md#word)'s properties `pos`, `xpos`, and `ufeats`. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| pos.cpu | bool | `False` (unset) | Set this flag to `True` to force the tagger to run on CPU. |
| pos.batch_size | int | 5000 | When annotating, this argument specifies the maximum number of words to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). This parameter should be set larger than the number of words in the longest sentence in your input document, or you might run into unexpected behaviors. |


## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/tagger.py#L21) of the POS/UFeats tagger.
