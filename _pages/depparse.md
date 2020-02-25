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

## Example Usage

The `depparse` processor depends on `tokenize`, `mwt`, `pos`, and `lemma`. After all these processors have been run, each `Sentence` in the output would have been parsed into Universal Dependencies (version 2) structure, where the governor index of each `word` can be accessed by `word.head`, and the dependency relation between the words `word.deprel`. Note that the governor index starts at 1 for actual words, and is 0 only when the word itself is the root of the tree. This index should be offset by 1 when looking for the govenor word in the sentence. Here is an example to access dependency parse information:

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma,depparse')
doc = nlp("Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie.")
print(*[f"index: {word.id.rjust(2)}\tword: {word.text.ljust(11)}\tgovernor index: {word.head}\tgovernor: {(doc.sentences[0].words[word.head-1].text if word.head > 0 else 'root').ljust(11)}\tdeprel: {word.deprel}" for word in doc.sentences[0].words], sep='\n')
```

This will generate the following output:

```
index:  1	word: Van        	governor index: 3	governor: grandit    	deprel: nsubj
index:  2	word: Gogh       	governor index: 1	governor: Van        	deprel: flat:name
index:  3	word: grandit    	governor index: 0	governor: root       	deprel: root
index:  4	word: à          	governor index: 6	governor: sein       	deprel: case
index:  5	word: le         	governor index: 6	governor: sein       	deprel: det
index:  6	word: sein       	governor index: 3	governor: grandit    	deprel: obl:mod
index:  7	word: d'         	governor index: 9	governor: famille    	deprel: case
index:  8	word: une        	governor index: 9	governor: famille    	deprel: det
index:  9	word: famille    	governor index: 6	governor: sein       	deprel: nmod
index: 10	word: de         	governor index: 13	governor: bourgeoisie	deprel: case
index: 11	word: l'         	governor index: 13	governor: bourgeoisie	deprel: det
index: 12	word: ancienne   	governor index: 13	governor: bourgeoisie	deprel: amod
index: 13	word: bourgeoisie	governor index: 9	governor: famille    	deprel: nmod
index: 14	word: .          	governor index: 3	governor: grandit    	deprel: punct
```

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/parser.py#L21) of the dependency parser.

