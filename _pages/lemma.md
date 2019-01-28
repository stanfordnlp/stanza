---
title: LemmaProcessor
keywords: lemma
permalink: '/lemma.html'
---

## Description

Generates the word lemmas for all tokens in the corpus.

| Property name | Processor class name | Generated Annotation |
| --- | --- | --- |
| lemma | LemmaProcessor | Perform [lemmatization](https://en.wikipedia.org/wiki/Lemmatisation) on a [`Word`](data_objects.md#word) using the `Word.text` and `Word.upos` value. The result can be accessed in `Word.lemma`. | 

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| lemma.cpu | bool | `False` | Set this flag to `True` to force the lemmatizer to run on CPU, even if GPU is available. |
| lemma.use_identity | bool | `False` | When this flag is used, an identity lemmatizer (see `models.identity_lemmatizer`) will be used instead of a statistical lemmatizer. This is useful when [`Word.lemma`] is required for languages such as Vietnamese, where the lemma is identical to the original word form. |
| lemma.batch_size | int | 50 | When annotating, this argument specifies the maximum number of words to batch for efficient processing. |
| lemma.ensemble_dict | bool | `True` | If set to `True`, the lemmatizer will ensemble a seq2seq model with the output from a dictionary-based lemmatizer, which yields improvements on many languages (see system description paper for more details). |
| lemma.dict_only | bool | `False` | If set to `True`, only a dictionary-based lemmatizer will be used. For languages such as Chinese, a dictionary-based lemmatizer is enough. |
| lemma.edit | bool | `True` | If set to `True`, use an edit classifier alongside the seq2seq lemmatizer. The edit classifier will predict "shortcut" operations such as "identical" or "lowercase", to make the lemmatization of long sequences more stable. |
| lemma.beam_size | int | 1 | Control the beam size used during decoding in the seq2seq lemmatizer. |
| lemma.max_dec_len | int | 50 | Control the maximum decoding character length in the seq2seq lemmatizer. The decoder will stop if this length is achieved and the end-of-sequence character is still not seen. |

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/lemmatizer.py#L22) of the lemmatizer.

