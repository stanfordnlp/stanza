---
layout: page
title: Part-of-Speech & Morphological Features
keywords: POS, part-of-speech, morphological features, POSProcessor
permalink: '/pos.html'
nav_order: 6
parent: Neural Pipeline
---

## Description

The Part-of-Speech (POS) & morphological features tagging module labels words with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html). This is jointly performed by the `POSProcessor` in Stanza, and can be invoked with the name `pos`.

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| pos | POSProcessor | tokenize, mwt | UPOS, XPOS, and UFeats annotations are accessible through [`Word`](data_objects.md#word)'s properties `pos`, `xpos`, and `ufeats`. | Labels tokens with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html). |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| pos_batch_size | int | 5000 | When annotating, this argument specifies the maximum number of words to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). This parameter should be set larger than the number of words in the longest sentence in your input document, or you might run into unexpected behaviors. |

## Example Usage

Running the [POSProcessor](pos.md) requires the [TokenizeProcessor](tokenize.md) and [MWTProcessor](mwt.md).
After the pipeline is run, the [`Document`](data_objects.md#document) will contain a list of [`Sentence`](data_objects.md#sentence)s, and the [`Sentence`](data_objects.md#sentence)s will contain lists of [`Word`](data_objects.md#word)s. The part-of-speech tags can be accessed via the `upos`(`pos`) and `xpos` fields of each [`Word`](data_objects.md#word), while the universal morphological features can be accessed via the `feats` field.

### Accessing POS and Morphological Feature for Word

Here is an example of tagging a piece of text and accessing part-of-speech and morphological features for each word:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')
```

As can be seen in the result, we can tell that the word `was` is a third-person auxiliary verb in the past tense from Stanza's analysis.

```
word: Barack    upos: PROPN     xpos: NNP       feats: Number=Sing
word: Obama     upos: PROPN     xpos: NNP       feats: Number=Sing
word: was       upos: AUX       xpos: VBD       feats: Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin
word: born      upos: VERB      xpos: VBN       feats: Tense=Past|VerbForm=Part|Voice=Pass
word: in        upos: ADP       xpos: IN        feats: _
word: Hawaii    upos: PROPN     xpos: NNP       feats: Number=Sing
word: .         upos: PUNCT     xpos: .         feats: _
```


## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/master/stanza/models/tagger.py#L21) of the POS/UFeats tagger.
