---
layout: page
title: Lemmatization
keywords: lemma, lemmatization, LemmaProcessor
permalink: '/lemma.html'
nav_order: 7
parent: Neural Pipeline
---

## Description

The lemmatization module recovers the lemma form for each input word. For example, the input sequence "I ate an apple" will be lemmatized into "I eat a apple". This type of word normalization is useful in many real-world applications. In Stanza, lemmatization is performed by the `LemmaProcessor` and can be invoked with the name `lemma`.

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| lemma | LemmaProcessor | tokenize, mwt, pos | Perform [lemmatization](https://en.wikipedia.org/wiki/Lemmatisation) on a [`Word`](data_objects.md#word) using the `Word.text` and `Word.upos` values. The result can be accessed as `Word.lemma`. | Generates the word lemmas for all words in the Document. |

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

Running the [LemmaProcessor](lemma.md) requires the [TokenizeProcessor](tokenize.md), [MWTProcessor](mwt.md), and [POSProcessor](pos.md).
After the pipeline is run, the [`Document`](data_objects.md#document) will contain a list of [`Sentence`](data_objects.md#sentence)s, and the [`Sentence`](data_objects.md#sentence)s will contain lists of [`Word`](data_objects.md#word)s.
The lemma information can be found in the `lemma` field of each [`Word`](data_objects.md#word).

### Accessing Lemma for Word

Here is an example of lemmatizing words in a sentence and accessing their lemmas afterwards:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'word: {word.text+" "}\tlemma: {word.lemma}' for sent in doc.sentences for word in sent.words], sep='\n')
```

As can be seen in the result, Stanza lemmatizes the word _was_ as _be_.

```
word: Barack    lemma: Barack
word: Obama     lemma: Obama
word: was       lemma: be
word: born      lemma: bear
word: in        lemma: in
word: Hawaii    lemma: Hawaii
word: .         lemma: .
```

### Improving the Lemmatizer by Providing Key-Value Dictionary

It is possible to improve the lemmatizer by providing a key-value dictionary. Lemmatizer will check it first and then use statistical model if the word is not in dictionary.

First, load your downloaded lemmatizer model. For English lemmatizer using `ewt` package, it can be found at `~/stanza_resources/en/lemma/ewt.pt`.

Second, customize two dictionaries: 1) `composite_dict` which maps `(word, pos)` to `lemma`; 2) `word_dict` which maps `word` to `lemma`. The lemmatizer will first check the composite dictionary, then word dictionary.

Finally, save your customized model and load it with `Stanza`.

Here is an example of customizing the lemmatizer by providing a key-value dictionary:

```python
# Load word_dict and composite_dict
import torch
model = torch.load('~/stanza_resources/en/lemma/ewt.pt', map_location='cpu')
word_dict, composite_dict = model['dicts']

# Customize your own dictionary
composite_dict[('myword', 'NOUN')] = 'mylemma'
word_dict['myword'] = 'mylemma'

# Save your model
torch.save(model, '~/stanza_resources/en/lemma/ewt_customized.pt')

# Load your customized model with Stanza
import stanza
nlp = stanza.Pipeline('en', package='ewt', processors='tokenize,pos,lemma', lemma_model_path='~/stanza_resources/en/lemma/ewt_customized.pt'
print(nlp('myword')) # Should get lemma 'mylemma'
```

As can be seen in the result, Stanza should lemmatize the word _myword_ as _mylemma_.

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/master/stanza/models/lemmatizer.py#L22) of the lemmatizer.

