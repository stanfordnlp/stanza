---
title: DepparseProcessor
keywords: depparse
permalink: '/depparse.html'
---

## Description

Provides an accurate syntactic dependency parser.

| Property name | Annotator class name | Generated Annotation |
| --- | --- | --- |
| depparse | DepparseProcessor | Determines the syntactic head of each word in a sentence and the dependency relation between the two words that are accessible through [`Word`](data_objects.md#word)'s `head` and `deprel` attributes. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| depparse_batch_size | int | 5000 | When annotating, this argument specifies the maximum number of words to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). This parameter should be set larger than the number of words in the longest sentence in your input document, or you might run into unexpected behaviors. |
| depparse_pretagged | bool | False | Assume the document is tokenized and pretagged. Only run dependency parsing on the document. |

## Example Usage

The `depparse` processor depends on `tokenize`, `mwt`, `pos`, and `lemma`. After all these processors have been run, each `Sentence` in the output would have been parsed into Universal Dependencies (version 2) structure, where the head index of each `word` can be accessed by `word.head`, and the dependency relation between the words `word.deprel`. Note that the head index starts at 1 for actual words, and is 0 only when the word itself is the root of the tree. This index should be offset by 1 when looking for the govenor word in the sentence. 

### Dependency Parsing

The code below shows an example of accessing the head word and dependency relation for each word:

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma,depparse')
doc = nlp('Nous avons atteint la fin du sentier.')
print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
```

This will generate the following output:

```
id: 1   word: Nous      head id: 3      head: atteint   deprel: nsubj
id: 2   word: avons     head id: 3      head: atteint   deprel: aux:tense
id: 3   word: atteint   head id: 0      head: root      deprel: root
id: 4   word: la        head id: 5      head: fin       deprel: det
id: 5   word: fin       head id: 3      head: atteint   deprel: obj
id: 6   word: de        head id: 8      head: sentier   deprel: case
id: 7   word: le        head id: 8      head: sentier   deprel: det
id: 8   word: sentier   head id: 5      head: fin       deprel: nmod
id: 9   word: .         head id: 3      head: atteint   deprel: punct
```

The head of the word `Nous` is `atteint` with the dependency relation `nsubj`.

### Start with Pretagged Document

Normally, the `depparse` processor depends on `tokenize`, `mwt`, `pos`, and `lemma`. However, you can skip the restriction and pass the pretagged document (with upos, xpos, feats, lemma) by setting `depparse_pretagged=True`.

The code below shows an example of dependency parsing with pretokenized and pretagged document:

```python
import stanfordnlp
from stanfordnlp.models.common.doc import Document

nlp = stanfordnlp.Pipeline(lang='en', processors='depparse', depparse_pretagged=True)
pretagged_doc = Document([[{'id': '1', 'text': 'Test', 'lemma': 'Test', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing'}, {'id': '2', 'text': 'sentence', 'lemma': 'sentence', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Number=Sing'}, {'id': '3', 'text': '.', 'lemma': '.', 'upos': 'PUNCT', 'xpos': '.'}]])
doc = nlp(pretagged_doc)
```

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/parser.py#L21) of the dependency parser.

