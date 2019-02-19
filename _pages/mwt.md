---
title: MWTProcessor
keywords: mwt
permalink: '/mwt.html'
---

## Description

Expands [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html) predicted by the [tokenizer](tokenize.md). 

| Property name | Processor class name | Generated Annotation |
| --- | --- | --- |
| mwt | MWTProcessor | Expands multi-word tokens into multiple words when they are predicted by the tokenizer. | 

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| mwt_batch_size | int | 50 | When annotating, this argument specifies the maximum number of words to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/mwt_expander.py#L22) of the MWT expander.
