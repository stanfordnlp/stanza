---
title: DepparseProcessor 
keywords: depparse
permalink: '/depparse.html'
---

## Description

Provides an accurate syntactic dependency parser.

| Property name | Annotator class name | Generated Annotation |
| --- | --- | --- |
| depparse | DepparseProcessor | Determines the syntactic head of each word in a sentence and the dependency relation between the two words that are accessible through [`Word`](data_objects.md#word)'s `governor` and `dependency_relation` attributes. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| depparse_batch_size | int | 5000 | When annotating, this argument specifies the maximum number of words to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). This parameter should be set larger than the number of words in the longest sentence in your input document, or you might run into unexpected behaviors. |


## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/parser.py#L21) of the dependency parser.

