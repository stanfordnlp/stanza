---
title: TokenizeProcessor
keywords: tokenize
permalink: '/tokenize.html'
---

## Description

Tokenizes the text and performs sentence segmentation. 

| Property name | Annotator class name | Generated Annotation |
| --- | --- | --- |
| tokenize | TokenizeProcessor | Segments a `Document` into `Sentence`s, each containing a list of `Token`s. This processor doesn't expand multi-word tokens (see the [MWT expander](/mwt_expander.html)). | 

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| tokenize.cpu | N/A | `False` (unset) | Set this flag to `True` to force the tokenizer to run on CPU. |
| tokenize.batch_size | int | 32 | When annotating, this argument specifies the maximum number of paragraphs to batch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |
| tokenize.whitespace | boolean | false | If set to true, separate |

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/tokenizer.py#L12) of the tokenizer.

Note that to train the tokenizer for Vietnamese, one would need to postprocess the character labels generated from the plain text file and the CoNLL-U file to form syllable-level labels, which is automatically handled if you are using the training scripts we provide.
