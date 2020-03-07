---
title: MWTProcessor
keywords: mwt
permalink: '/mwt.html'
---

## Description

Expands [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html) predicted by the [TokenizeProcessor](tokenize.md).

<div class="alert alert-warning" role="alert">
Note: Only languages with <a href="https://universaldependencies.org/u/overview/tokenization.html">multi-word tokens (MWT)</a> require MWTProcessor.
</div>

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- | 
| mwt | MWTProcessor | tokenize | Expands multi-word tokens into multiple words when they are predicted by the tokenizer. | Expands [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html) predicted by the [TokenizeProcessor](tokenize.md). |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| mwt_batch_size | int | 50 | When annotating, this argument specifies the maximum number of words to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |

## Example Usage

The [MWTProcessor](mwt.md) processor only requires [TokenizeProcessor](tokenize.md). After these two processors have run, the [`Sentence`](data_objects.md#sentence)s will have lists of [`Token`](data_objects.md#token)s and corresponding [`Word`](data_objects.md#word)s based on the multi-word-token expander model.  The list of tokens for sentence `sent` can be accessed with `sent.tokens`.  The list of words for sentence `sent` can be accessed with `sent.words`.  The list of words for a token `token` can be accessed with `token.words`.  

### Access Syntactic Words for Multi-Word Token

The code below shows an example of accessing syntactic words for each token in the first sentence:

```python
import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt')
doc = nlp('Nous avons atteint la fin du sentier.')
for token in doc.sentences[0].tokens:
    print(f'token: {token.text}\twords: {", ".join([word.text for word in token.words])}')
```

This code will generate the following output:

```
token: Nous     words: Nous
token: avons    words: avons
token: atteint  words: atteint
token: la       words: la
token: fin      words: fin
token: du       words: de, le
token: sentier  words: sentier
token: .        words: .
```

The multi-word token `du` is expanded to two syntactic words `de` and `le`.

### Access Parent Token for Word

The code below shows an example of accessing parent token for each word in the first sentence:

```python
import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt')
doc = nlp('Nous avons atteint la fin du sentier.')
for word in doc.sentences[0].words:
    print(f'word: {word.text}\tparent token: {word.parent.text}')
```

This code will generate the following output:

```
word: Nous      parent token: Nous
word: avons     parent token: avons
word: atteint   parent token: atteint
word: la        parent token: la
word: fin       parent token: fin
word: de        parent token: du
word: le        parent token: du
word: sentier   parent token: sentier
word: .         parent token: .
```

Words `de` and `le` have the same parent token `du`.

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/master/stanza/models/mwt_expander.py#L22) of the MWT expander.
