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
| lemma_use_identity | bool | `False` | When this flag is used, an identity lemmatizer (see `models.identity_lemmatizer`) will be used instead of a statistical lemmatizer. This is useful when [`Word.lemma`] is required for languages such as Vietnamese, where the lemma is identical to the original word form. |
| lemma_batch_size | int | 50 | When annotating, this argument specifies the maximum number of words to batch for efficient processing. |
| lemma_ensemble_dict | bool | `True` | If set to `True`, the lemmatizer will ensemble a seq2seq model with the output from a dictionary-based lemmatizer, which yields improvements on many languages (see system description paper for more details). |
| lemma_dict_only | bool | `False` | If set to `True`, only a dictionary-based lemmatizer will be used. For languages such as Chinese, a dictionary-based lemmatizer is enough. |
| lemma_edit | bool | `True` | If set to `True`, use an edit classifier alongside the seq2seq lemmatizer. The edit classifier will predict "shortcut" operations such as "identical" or "lowercase", to make the lemmatization of long sequences more stable. |
| lemma_beam_size | int | 1 | Control the beam size used during decoding in the seq2seq lemmatizer. |
| lemma_max_dec_len | int | 50 | Control the maximum decoding character length in the seq2seq lemmatizer. The decoder will stop if this length is achieved and the end-of-sequence character is still not seen. |

## Example Usage

Running the lemmatizer requires tokenization, multi-word expansion and part-of-speech tagging.
After the pipeline is run, the document will contain a list of sentences, and the sentences will contain lists of words.
The lemma information can be found in the `lemma` field of each word.

### Lemmatization

The code below shows an example of accessing lemma for each word:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'word: {word.text+" "}\tlemma: {word.lemma}' for sent in doc.sentences for word in sent.words], sep='\n')
```

This code will generate the following output:

```
word: Barack    lemma: Barack
word: Obama     lemma: Obama
word: was       lemma: be
word: born      lemma: bear
word: in        lemma: in
word: Hawaii    lemma: Hawaii
word: .         lemma: .
```

The lemma of the word `was` is `be`.

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/master/stanza/models/lemmatizer.py#L22) of the lemmatizer.

