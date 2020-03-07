---
title: NERProcessor 
keywords: ner
permalink: '/ner.html'
---

## Description

Recognize named entities for all token spans in the corpus.

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- | 
| ner | NERProcessor | tokenize, mwt | Named entities accessible through [`Document`](data_objects.md#document) or [`Sentence`](data_objects.md#sentence)'s properties `entities` or `ents`. Token-level NER tags accessible through [`Token`](data_objects.md#token)'s properties `ner`. | Recognize named entities for all token spans in the corpus. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| ner_batch_size | int | 32 | When annotating, this argument specifies the maximum number of sentences to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |


## Example Usage

Running the [NERProcessor](ner.md) simply requires [TokenizeProcessor](tokenize.md). After the pipeline is run, the [`Document`](data_objects.md#document) will contain a list of [`Sentence`](data_objects.md#sentence)s, and the [`Sentence`](data_objects.md#sentence)s will contain lists of [`Token`](data_objects.md#token)s. 
Named entities can be accessed through [`Document`](data_objects.md#document) or [`Sentence`](data_objects.md#sentence)'s properties `entities` or `ents`.
Alternatively, token-level NER tags can be accessed via the `ner` fields of [`Token`](data_objects.md#token).

### Named Entity Recognition

The code below shows an example of accessing the named entities in the document:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
```

Alternatively, you can access the named entities in each sentence of the document. 

The equivalent of our example above would be:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')
```

These codes will generate the following output:

```
entity: Barack Obama    type: PERSON
entity: Hawaii          type: GPE
```

The span `Barack Obama` is a person entity, while the span `Hawaii` is a geopolitical entity.

### Access Token-level NER Tags

The code below shows an example of accessing the ner tags for each token:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'token: {token.text}\tner: {token.ner}' for sent in doc.sentences for token in sent.tokens], sep='\n')
```

This code will generate the following output:

```
token: Barack   ner: B-PERSON
token: Obama    ner: E-PERSON
token: was      ner: O
token: born     ner: O
token: in       ner: O
token: Hawaii   ner: S-GPE
token: .        ner: O
```

The token `Barack` is the beginning of the person entity, while the token `Obama` is the end of the person entity.

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/master/stanza/models/ner_tagger.py#L32) of the NER tagger.