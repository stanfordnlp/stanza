---
layout: page
title: Multi-Word Token (MWT) Expansion
keywords: mwt, multi-word token expansion, MWTProcessor
permalink: '/mwt.html'
nav_order: 5
parent: Neural Pipeline
---

## Description

The Multi-Word Token (MWT) expansion module can expand a raw token into multiple [syntactic words](https://universaldependencies.org/u/overview/tokenization.html), which makes it easier to carry out Universal Dependencies analysis in some languages. This was handled by the `MWTProcessor` in Stanza, and can be invoked with the name `mwt`. The token upon which an expansion will be performed is predicted by the `TokenizeProcessor`, before the invocation of the `MWTProcessor`.

For more details on why MWT is necessary for Universal Dependencies analysis, please visit the [UD tokenization page](https://universaldependencies.org/u/overview/tokenization.html).

{% include alerts.html %}
{{ note }}
Only languages with <a href='https://universaldependencies.org/u/overview/tokenization.html'>multi-word tokens (MWT)</a>, such as German or French, require MWTProcessor; other languages, such as English or Chinese, do not support this processor in the pipeline.
{{ end }}

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| mwt | MWTProcessor | tokenize | Expands multi-word tokens (MWTs) into multiple words when they are predicted by the tokenizer. Each [`Token`](data_objects.md#token) will correspond to one or more [`Word`](data_objects.md#word)s after tokenization and MWT expansion. | Expands [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html) predicted by the [TokenizeProcessor](tokenize.md). This is only applicable to some languages. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| mwt_batch_size | int | 50 | When annotating, this argument specifies the maximum number of words to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |

## Example Usage

The [MWTProcessor](mwt.md) processor only requires [TokenizeProcessor](tokenize.md) to be run before it. After these two processors have processed the text, the [`Sentence`](data_objects.md#sentence)s will have lists of [`Token`](data_objects.md#token)s and corresponding syntactic [`Word`](data_objects.md#word)s based on the multi-word-token expander model.  The list of tokens for a sentence `sent` can be accessed with `sent.tokens`, and its list of words with `sent.words`. Similarly, the list of words for a token `token` can be accessed with `token.words`.

### Accessing Syntactic Words for Multi-Word Token

Here is an example of a piece of text in French that requires multi-word token expansion, and how to access the underlying words of these multi-word tokens:

```python
import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt')
doc = nlp('Nous avons atteint la fin du sentier.')
for token in doc.sentences[0].tokens:
    print(f'token: {token.text}\twords: {", ".join([word.text for word in token.words])}')
```

As a result of running this code, we see that the word _du_ is expanded into its underlying syntactic words, _de_ and _le_.

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

### Accessing Parent Token for Word

When performing word-level annotations and processing, it might sometimes be useful to access the token a given word is derived from, so that we can access its character offsets, among other things, that are associated with the token. Here is an example of how to do that with [`Word`](data_object.md#word)'s `parent` property with the same sentence we just saw:

```python
import stanza

nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt')
doc = nlp('Nous avons atteint la fin du sentier.')
for word in doc.sentences[0].words:
    print(f'word: {word.text}\tparent token: {word.parent.text}')
```

As one can see in the result below, Words `de` and `le` have the same parent token `du`.

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

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/main/stanza/models/mwt_expander.py#L22) of the MWT expander.

## Resplitting tokens with MWT

In some circumstances, you may want to use the MWT processor to
resplit known tokens.  For example, we had an instance where an
Italian dataset included token boundaries, but the words were not
separated into clitics.  For this, we provide a utility function

`[resplit_mwt](https://github.com/stanfordnlp/stanza/blob/5b3c8b31a9cd238127d6db873b825832051ec3df/stanza/models/mwt/utils.py#L7)`

This function takes a list of list of string, representing the known
token boundaries, and a `Pipeline` with a minimum of a tokenizer and
MWT processor, and retokenizes the text, returning a Stanza
`Document`.  An example usage
[is in the unit test for this method](https://github.com/stanfordnlp/stanza/blob/5b3c8b31a9cd238127d6db873b825832051ec3df/stanza/tests/mwt/test_utils.py#L22)

If further processing is needed, this `Document` can be passed to
another pipeline which uses the
[`tokenize_pretokenized`](https://stanfordnlp.github.io/stanza/tokenize.html#start-with-pretokenized-text)
flag, in which case the second pipeline will respect the tokenization
and MWT boundaries in the `Document`.