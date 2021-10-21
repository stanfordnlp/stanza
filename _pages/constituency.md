---
layout: page
title: Sentiment Analysis
keywords: sentiment, classifier
permalink: '/sentiment.html'
nav_order: 10
parent: Neural Pipeline
---

## Description

Constituency parsing is added to the stanza pipeline by using [a shift-reduce parser](https://aclanthology.org/Q17-1029/).

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| constituency | ConstituencyProcessor | tokenize, mwt, pos | `constituency` | Adds the `constituency` annotation to each [`Sentence`](data_objects.md#sentence) in the `Document` |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| 'model_path' | string | depends on the language | Where to load the model. |
| 'pretrain_path' | string | depends on the language | Which set of pretrained word vectors to use. Can be changed for existing models, but this is not recommended, as the models are trained to work specifically with one set of word vectors. |

## Example Usage

The `ConstituencyProcessor` adds a constituency / phrase structure
[parse tree](data_objects.md#parsetree) to each [`Sentence`](data_objects.md#sentence).

Bracket types are dependent on the treebank; for example, the PTB
model using the PTB bracket types.  Custom models could support any
set of labels as long as you have training data.

### Simple code example

import stanza

```
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
doc = nlp('This is a test')
for sentence in doc.sentences:
    print(sentence.constituency)
```

The output produced (aside from logging) will be:

```
(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))
```

The tree can be programmatically accessed.  Note that the layer under the root has two children, one for the `NP This` and one for the `VP is a test`.

```
>>> tree = doc.sentences[0].constituency
>>> tree.label
'ROOT'
>>> tree.children
[(S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test))))]
>>> tree.children[0].children
[(NP (DT This)), (VP (VBZ is) (NP (DT a) (NN test)))]
```


## Available models

There is currently one model available for constituency parsing.

As of Stanza 1.3.0, there is an English model trained on PTB.  It achieves a test score of 91.5 using the inorder transition scheme.

Unfortunately, there is a bug in the model where, if trees with `)` in
the tokens are converted to text, the resulting trees will not be
properly bracketed.  This has been fixed in the dev branch.  A new
release with updated models should be available in mid-November.

Also coming soon in 1.3.1 are the following improvements:

- Italian model trained on the [Turin treebank](http://www.di.unito.it/~tutreeb/treebanks.html)
- Integration with the pretrained charlm.  This improves the PTB test score to 92.3
- More languages dependending on treebank availability and interest.  (Vietnamese, for example.)

