---
title: POSProcessor 
keywords: pos
permalink: '/pos.html'
---

## Description

Labels tokens with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html).

| Property name | Annotator class name | Generated Annotation |
| --- | --- | --- |
| pos | POSProcessor | UPOS, XPOS, and UFeats annotations accessible through [`Word`](data_objects.md#word)'s properties `pos`, `xpos`, and `ufeats`. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| pos_batch_size | int | 5000 | When annotating, this argument specifies the maximum number of words to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). This parameter should be set larger than the number of words in the longest sentence in your input document, or you might run into unexpected behaviors. |

## Example Usage

Running the part of speech tagger simply requires tokenization and multi-word expansion.  So the pipeline
can be run with `tokenize,mwt,pos` as the list of processors.  After the pipeline is run, the document will 
contain a list of sentences, and the sentences will contain lists of words. The part-of-speech tags can 
be accessed via the `upos` and `xpos` fields.

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')
doc = nlp("Barack Obama was born in Hawaii.")
print(*[f'word: {word.text+" "}\tupos: {word.upos}\txpos: {word.xpos}' for sent in doc.sentences for word in sent.words], sep='\n')
```

This code will generate the following output:

```
word: Barack 	upos: PROPN	xpos: NNP
word: Obama 	upos: PROPN	xpos: NNP
word: was 	upos: AUX	xpos: VBD
word: born 	upos: VERB	xpos: VBN
word: in 	upos: ADP	xpos: IN
word: Hawaii 	upos: PROPN	xpos: NNP
word: . 	upos: PUNCT	xpos: .
```

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/tagger.py#L21) of the POS/UFeats tagger.
