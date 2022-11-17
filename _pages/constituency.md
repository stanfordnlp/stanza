---
layout: page
title: Constituency Parser
keywords: constituency
permalink: '/constituency.html'
nav_order: 9
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

```python
import stanza

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

As of Stanza 1.4.0, charlm has been added by default to each of the
conparse models.  This improves accuracy around 1.0 F1 when trained
for a long time.  The currently released models were trained on 250
iterations of 5000 trees each, so for languages with large datasets
such as English, there may have been room to improve further.

We also release a set of models which incorporate HuggingFace
transformer models such as Bert or Roberta.  This significantly
increases the scores for the constituency parser.

Bert models can be used by setting the package parameter when creating
a pipeline:

```python
pipe = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', package={'constituency': 'wsj_bert'})
```

Note that the following scores are slight underestimates.  They use the CoreNLP scorer, which gives scores slightly lower than the evalb script.

| Language | Dataset | Base score | Transformer score | Notes |
| --- | --- | --- | --- |
| Chinese | CTB7 | 86.85 | 90.98 | Future work: update to later versions of CTB |
| Danish | [Arboretum](http://catalog.elra.info/en-us/repository/browse/ELRA-W0084/) | 82.7 | 83.45 | [Non-projective constituents are rearranged](https://github.com/stanfordnlp/stanza/blob/main/stanza/utils/datasets/constituency/convert_arboretum.py) |
| English | PTB | 93.21 | 95.64 | with NMLs and separated dashes |
| English | PTB3 | --- | 95.72 | |
| Italian | [Turin](http://www.di.unito.it/~tutreeb/treebanks.html) | 89.42 | 92.76 | Test scores are on [Evalita](http://www.di.unito.it/~tutreeb/evalita-parsingtask-11.html) |
| Italian | VIT | 78.52 | 82.43 | Split based on UD VIT (some trees dropped) |
| Japanese | [ALT](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/) | 91.03 | --- | Transformers were not used - required a separate tokenizer |
| Portuguese | [Cintil](https://hdl.handle.net/21.11129/0000-000B-D2FE-A) | 90.98 | 93.61 | |
| Spanish | AnCora + LDC | ??? | ??? | Compared against a combination of the test sets |
| Turkish | Starlang | 73.04 | 75.7 | |

As of Stanza 1.3.0, there was an English model trained on PTB.
It achieved a test score of 91.5 using the inorder transition scheme.


## Other links

Cintil can also be purchased from [ELRA](https://catalogue.elra.info/en-us/repository/browse/ELRA-W0055/)

## Citations

Silva, João, António Branco, Sérgio Castro and Ruben Reis, 2010, "Out-of-the-Box Robust Parsing of Portuguese". In Proceedings of the 9th International Conference on the Computational Processing of Portuguese (PROPOR2010), Lecture Notes in Artificial Intelligence, 6001, Berlin, Springer, pp.75–85.

