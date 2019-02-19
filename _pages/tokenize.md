---
title: TokenizeProcessor
keywords: tokenize
permalink: '/tokenize.html'
---

## Description

Tokenizes the text and performs sentence segmentation. 

| Property name | Processor class name | Generated Annotation |
| --- | --- | --- |
| tokenize | TokenizeProcessor | Segments a [`Document`](data_objects.md#document) into [`Sentence`](data_objects.md#sentence)s, each containing a list of [`Token`](data_objects.md#token)s. This processor also predicts which tokens are multi-word tokens, but leaves expanding them to the [MWT expander](mwt.md). | 

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| tokenize_batch_size | int | 32 | When annotating, this argument specifies the maximum number of paragraphs to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |
| tokenize_pretokenized | bool | False | Assume the text is tokenized by white space and sentence split by newline.  Do not run a model. |

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/tokenizer.py#L12) of the tokenizer.

Note that to train the tokenizer for Vietnamese, one would need to postprocess the character labels generated from the plain text file and the CoNLL-U file to form syllable-level labels, which is automatically handled if you are using the training scripts we provide.
