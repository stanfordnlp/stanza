---
title: NERProcessor 
keywords: ner
permalink: '/ner.html'
---

## Description

Recognize named entities for all token spans in the corpus.

| Property name | Annotator class name | Generated Annotation |
| --- | --- | --- |
| ner | NERProcessor | Named entities accessible through [`Document`](data_objects.md#document) or [`Sentence`](data_objects.md#sentence)'s properties `entities` or `ents`. Token-level NER tags accessible through [`Token`](data_objects.md#token)'s properties `ner`. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| ner_batch_size | int | 32 | When annotating, this argument specifies the maximum number of sentences to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |


## Example Usage

Running the named entity tagger simply requires tokenization.  So the pipeline
can be run with `tokenize` as the list of processors.  After the pipeline is run, the document will 
contain a list of sentences, and the sentences will contain lists of named entities. 
Named entities can be accessed through [`Document`](data_objects.md#document) or [`Sentence`](data_objects.md#sentence)'s properties `entities` or `ents`.
Token-level NER tags can be accessed via the `ner` fields of [`Token`](data_objects.md#token).

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline(lang="en", processors="tokenize,ner")
doc = nlp("Barack Obama was born in Hawaii.")
print(*[f'entity: {ent.text+" "}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
print('')
print(*[f'token: {token.text+" "}\tner: {token.ner}' for sent in doc.sentences for token in sent.tokens], sep='\n')
```

This code will generate the following output:

```
entity: Barack Obama 	type: PERSON
entity: Hawaii 	type: GPE
```

```
token: Barack 	ner: B-PERSON
token: Obama 	ner: E-PERSON
token: was 	ner: O
token: born 	ner: O
token: in 	ner: O
token: Hawaii 	ner: S-GPE
token: . 	ner: O
```

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/ner_tagger.py#L32) of the NER tagger.