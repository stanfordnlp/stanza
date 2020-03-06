---
title: Data Conversion
keywords: data conversion
permalink: '/data_conversion.html'
---

This page describes the data conversion for Stanza, and how can we seamlessly convert between [`Document`](data_objects.md#document) and [`CoNLL`](https://universaldependencies.org/format.html) through internal python object. We show four examples that represent exactly the same document.

## Document to Python Object

A [`Document`](data_objects.md#document) instance will be returned after annotated by the [`Pipeline`](data_objects.md#pipeline). 

The code below shows an example of converting [`Document`](data_objects.md#document) to python native object:

```python
import stanza

nlp = stanza.Pipeline('en', processors='tokenize,pos')
doc = nlp('Test sentence.') # doc is class Document
dicts = doc.to_dict() # dicts is List[List[Dict]], representing each token / word in each sentence in the document
```

## Python Object to Document

A [`Document`](data_objects.md#document) can be instanciated with python native object. And it can be passed to the [`Pipeline`](data_objects.md#pipeline) for further annotations.

The code below shows an example of converting python native object to [`Document`]:

```python
from stanza.models.common.doc import Document

dicts = [[{'id': '1', 'text': 'Test', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=0|end_char=4'}, {'id': '2', 'text': 'sentence', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=5|end_char=13'}, {'id': '3', 'text': '.', 'upos': 'PUNCT', 'xpos': '.', 'misc': 'start_char=13|end_char=14'}]] # dicts is List[List[Dict]], representing each token / word in each sentence in the document
doc = Document(dicts) # doc is class Document
```

## CoNLL to Python Object

[`CoNLL`](https://universaldependencies.org/format.html) is a widely-used format for universal dependencies. 

The code below shows an example of converting [`CoNLL`](https://universaldependencies.org/format.html) to python native object.

```python
from stanza.utils.conll import CoNLL

conll = [[['1', 'Test', '_', 'NOUN', 'NN', 'Number=Sing', '0', '_', '_', 'start_char=0|end_char=4'], ['2', 'sentence', '_', 'NOUN', 'NN', 'Number=Sing', '1', '_', '_', 'start_char=5|end_char=13'], ['3', '.', '_', 'PUNCT', '.', '_', '2', '_', '_', 'start_char=13|end_char=14']]] # conll is List[List[List]], representing each token / word in each sentence in the document
dicts = CoNLL.convert_conll(conll) # dicts is List[List[Dict]], representing each token / word in each sentence in the document
```

## Python Object to CoNLL

[`CoNLL`](https://universaldependencies.org/format.html) is a widely-used format for universal dependencies. 

The code below shows an example of converting python native object to [`CoNLL`](https://universaldependencies.org/format.html).

```python
from stanza.utils.conll import CoNLL

dicts = [[{'id': '1', 'text': 'Test', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=0|end_char=4'}, {'id': '2', 'text': 'sentence', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=5|end_char=13'}, {'id': '3', 'text': '.', 'upos': 'PUNCT', 'xpos': '.', 'misc': 'start_char=13|end_char=14'}]] # dicts is List[List[Dict]], representing each token / word in each sentence in the document
conll = CoNLL.convert_dict(dicts) # conll is List[List[List]], representing each token / word in each sentence in the document
```