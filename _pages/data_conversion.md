---
layout: page
title: Data Conversion
keywords: data conversion
permalink: '/data_conversion.html'
nav_order: 3
parent: Neural Pipeline
---

This page describes how to seamlessly convert between Stanza's [`Document`](data_objects.md#document), the [CoNLL-U format](https://universaldependencies.org/format.html), and native Python objects. We show four examples that represent exactly the same document.

## Document to Python Object

A [`Document`](data_objects.md#document) instance will be returned after text is annotated by the [`Pipeline`](data_objects.md#pipeline).

Here's how we can convert this [`Document`](data_objects.md#document) object to a native Python object:

```python
import stanza

nlp = stanza.Pipeline('en', processors='tokenize,pos')
doc = nlp('Test sentence.') # doc is class Document
dicts = doc.to_dict() # dicts is List[List[Dict]], representing each token / word in each sentence in the document
```

## Python Object to Document

A [`Document`](data_objects.md#document) can also be instanciated with a native Python object, and then passed to the [`Pipeline`](data_objects.md#pipeline) for further annotations.

The code below shows an example of converting python native object to [`Document`]:

```python
from stanza.models.common.doc import Document

dicts = [[{'id': 1, 'text': 'Test', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=0|end_char=4'}, {'id': 2, 'text': 'sentence', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=5|end_char=13'}, {'id': 3, 'text': '.', 'upos': 'PUNCT', 'xpos': '.', 'misc': 'start_char=13|end_char=14'}]] # dicts is List[List[Dict]], representing each token / word in each sentence in the document
doc = Document(dicts) # doc is class Document
```

## CoNLL to Python Object

[CoNLL-U](https://universaldependencies.org/format.html) is a widely-used format for universal dependencies.
Here is an example of converting from a CoNLL-U format to a native Python object with the help of Stanza:

```python
from stanza.utils.conll import CoNLL

conll = [[['1', 'Test', '_', 'NOUN', 'NN', 'Number=Sing', '0', '_', '_', 'start_char=0|end_char=4'], ['2', 'sentence', '_', 'NOUN', 'NN', 'Number=Sing', '1', '_', '_', 'start_char=5|end_char=13'], ['3', '.', '_', 'PUNCT', '.', '_', '2', '_', '_', 'start_char=13|end_char=14']]] # conll is List[List[List]], representing each token / word in each sentence in the document
dicts = CoNLL.convert_conll(conll) # dicts is List[List[Dict]], representing each token / word in each sentence in the document
```

## Python Object to CoNLL

It might sometimes be desirable to output or process data in the [CoNLL-U format](https://universaldependencies.org/format.html) after text has been annotated by Stanza. Here is an example of converting a native Python object to the CoNLL-U format:

```python
from stanza.utils.conll import CoNLL

dicts = [[{'id': '1', 'text': 'Test', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=0|end_char=4'}, {'id': '2', 'text': 'sentence', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing', 'misc': 'start_char=5|end_char=13'}, {'id': '3', 'text': '.', 'upos': 'PUNCT', 'xpos': '.', 'misc': 'start_char=13|end_char=14'}]] # dicts is List[List[Dict]], representing each token / word in each sentence in the document
conll = CoNLL.convert_dict(dicts) # conll is List[List[List]], representing each token / word in each sentence in the document
```

{% include alerts.html %}
{{ note }}
{{ "`convert_dict` is now marked as deprecated, as internally we use the Document object everywhere.  If you have a use case where you need it, please let us know!" | markdownify }}
{{ end }}

## CoNLL to Document

New in v1.2.1
{: .label .label-green }

There is a mechanism for converting CoNLL files directly to a Stanza Document:

```python
from stanza.utils.conll import CoNLL
doc = CoNLL.conll2doc("extern_data/ud2/ud-treebanks-v2.7/UD_Italian-ISDT/it_isdt-ud-train.conllu")
```

This can be used with a [pipeline on pretokenized text](tokenize.md#start-with-pretokenized-text) to reprocess parts of the document.  For example, this will reprocess the tags in the ISDT dataset.  Note that `nlp(doc)` alters the `doc` object in place in this example.

```python
doc = CoNLL.conll2doc("extern_data/ud2/ud-treebanks-v2.7/UD_Italian-ISDT/it_isdt-ud-train.conllu")
nlp = stanza.Pipeline(lang='it', processors='tokenize,pos', tokenize_pretokenized=True)
doc = nlp(doc)
```

## Document to CoNLL

New in v1.2.1
{: .label .label-green }

There is a corresponding mechanism for writing back the document:

```python
CoNLL.write_doc2conll(doc2, "output.conllu")
```
