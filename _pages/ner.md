---
layout: page
title: Named Entity Recognition
keywords: ner, named entity recognition, NERProcessor
permalink: '/ner.html'
nav_order: 10
parent: Neural Pipeline
---

## Description

The named entity recognition (NER) module recognizes mention spans of a particular entity type (e.g., Person or Organization) in the input sentence. NER is widely used in many NLP applications such as information extraction or question answering systems. In Stanza, NER is performed by the `NERProcessor` and can be invoked by the name `ner`.

{% include alerts.html %}
{{ note }}
{{ "The NERProcessor currently supports 23 languages. All supported languages along with their training datasets can be found [here](ner_models.md)." | markdownify }}
{{ end }}

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| ner | NERProcessor | tokenize, mwt | Named entities accessible through [`Document`](data_objects.md#document) or [`Sentence`](data_objects.md#sentence)'s properties `entities` or `ents`. Token-level NER tags accessible through [`Token`](data_objects.md#token)'s properties `ner`. | Recognize named entities for all token spans in the corpus. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| ner_batch_size | int | 32 | When annotating, this argument specifies the maximum number of sentences to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |
| ner_pretrain_path | str | model-specific | Where to get the pretrained word embedding.  If you trained your own NER model with a different pretrain from the default, you will need to set this flag to use the model. |

## Example Usage

Running the [NERProcessor](ner.md) simply requires the [TokenizeProcessor](tokenize.md). After the pipeline is run, the [`Document`](data_objects.md#document) will contain a list of [`Sentence`](data_objects.md#sentence)s, and the [`Sentence`](data_objects.md#sentence)s will contain lists of [`Token`](data_objects.md#token)s.
Named entities can be accessed through [`Document`](data_objects.md#document) or [`Sentence`](data_objects.md#sentence)'s properties `entities` or `ents`.
Alternatively, token-level NER tags can be accessed via the `ner` fields of [`Token`](data_objects.md#token).

### Accessing Named Entities for Sentence and Document

Here is an example of performing named entity recognition for a piece of text and accessing the named entities in the entire document:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
doc = nlp("Chris Manning teaches at Stanford University. He lives in the Bay Area.")
print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
```

Instead of accessing entities in the entire document, you can also access the named entities in each sentence of the document. The following example provides an identical result from the one above, by accessing entities from sentences instead of the entire document:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
doc = nlp("Chris Manning teaches at Stanford University. He lives in the Bay Area.")
print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')
```

As can be seen in the output, Stanza correctly identifies that _Chris Manning_ is a person, _Stanford University_ an organization, and _the Bay Area_ is a location.

```
entity: Chris Manning	type: PERSON
entity: Stanford University	type: ORG
entity: the Bay Area	type: LOC
```


### Accessing Named Entity Recogition (NER) Tags for Token

It might sometimes be useful to access the BIOES NER tags for each token, and here is an example how:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
doc = nlp("Chris Manning teaches at Stanford University. He lives in the Bay Area.")
print(*[f'token: {token.text}\tner: {token.ner}' for sent in doc.sentences for token in sent.tokens], sep='\n')
```

The result is the BIOES representation of the entities we saw above

```
token: Chris	ner: B-PERSON
token: Manning	ner: E-PERSON
token: teaches	ner: O
token: at	ner: O
token: Stanford	ner: B-ORG
token: University	ner: E-ORG
token: .	ner: O
token: He	ner: O
token: lives	ner: O
token: in	ner: O
token: the	ner: B-LOC
token: Bay	ner: I-LOC
token: Area	ner: E-LOC
token: .	ner: O
```

### Using multiple models

New in v1.4.0
{: .label .label-green }

When creating the pipeline, it is possible to use multiple NER models
at once by specifying a list in the package dict.  Here is a brief example:

```python
import stanza
pipe = stanza.Pipeline("en", processors="tokenize,ner", package={"ner": ["ncbi_disease", "ontonotes"]})
doc = pipe("John Bauer works at Stanford and has hip arthritis.  He works for Chris Manning")
print(doc.ents)
```

Output:

```
[{
  "text": "John Bauer",
  "type": "PERSON",
  "start_char": 0,
  "end_char": 10
}, {
  "text": "Stanford",
  "type": "ORG",
  "start_char": 20,
  "end_char": 28
}, {
  "text": "hip arthritis",
  "type": "DISEASE",
  "start_char": 37,
  "end_char": 50
}, {
  "text": "Chris Manning",
  "type": "PERSON",
  "start_char": 66,
  "end_char": 79
}]
```

Furthermore, the `multi_ner` field will have the outputs of each NER model in order.

```python
# results truncated for legibility
# note that the token ids start from 1, not 0
print(doc.sentences[0].tokens[0:2])
print(doc.sentences[0].tokens[8:10])
```

Output:

```
[[
  {
    "id": 1,
    "text": "John",
    "start_char": 0,
    "end_char": 4,
    "ner": "B-PERSON",
    "multi_ner": [
      "O",
      "B-PERSON"
    ]
  }
], [
  {
    "id": 2,
    "text": "Bauer",
    "start_char": 5,
    "end_char": 10,
    "ner": "E-PERSON",
    "multi_ner": [
      "O",
      "E-PERSON"
    ]
  }
]]
[[
  {
    "id": 8,
    "text": "hip",
    "start_char": 37,
    "end_char": 40,
    "ner": "B-DISEASE",
    "multi_ner": [
      "B-DISEASE",
      "O"
    ]
  }
], [
  {
    "id": 9,
    "text": "arthritis",
    "start_char": 41,
    "end_char": 50,
    "ner": "E-DISEASE",
    "multi_ner": [
      "E-DISEASE",
      "O"
    ]
  }
]]
```



## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/main/stanza/models/ner_tagger.py#L32) of the NER tagger.
