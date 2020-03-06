---
title: POSProcessor 
keywords: pos
permalink: '/pos.html'
---

## Description

Labels tokens with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html).

| Property name | Annotator class name | Generated Annotation |
| --- | --- | --- |
| pos | POSProcessor | UPOS, XPOS, and UFeats annotations accessible through [`Word`](data_objects.md#word)'s properties `upos`(`pos`), `xpos`, and `feats`. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| pos_batch_size | int | 5000 | When annotating, this argument specifies the maximum number of words to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). This parameter should be set larger than the number of words in the longest sentence in your input document, or you might run into unexpected behaviors. |

## Example Usage

Running the part-of-speech tagger requires tokenization and multi-word expansion. 
After the pipeline is run, the document will contain a list of sentences, and the sentences will contain lists of words. The part-of-speech tags can 
be accessed via the `upos`(`pos`) and `xpos` fields of each `word`, while the universal morphological features can be accessed via the `feats` field.

### POS and Morphological Feature Tagging

The code below shows an example of accessing part-of-speech and morphological features for each word:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')
```

This code will generate the following output:

```
word: Barack    upos: PROPN     xpos: NNP       feats: Number=Sing
word: Obama     upos: PROPN     xpos: NNP       feats: Number=Sing
word: was       upos: AUX       xpos: VBD       feats: Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin
word: born      upos: VERB      xpos: VBN       feats: Tense=Past|VerbForm=Part|Voice=Pass
word: in        upos: ADP       xpos: IN        feats: _
word: Hawaii    upos: PROPN     xpos: NNP       feats: Number=Sing
word: .         upos: PUNCT     xpos: .         feats: _
```

The word `was` is an auxiliary verb in the past tense.

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/master/stanza/models/tagger.py#L21) of the POS/UFeats tagger.
