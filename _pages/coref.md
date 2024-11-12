---
layout: page
title: Coreference
keywords: coref, coreference
permalink: '/coref.html'
nav_order: 13
parent: Neural Pipeline
---

## Overview

Stanza 1.7.0 introduces a coreference model.  The model is
[Conjunction-Aware Word-level Coreference Resolution](https://arxiv.org/abs/2310.06165),
by Karel D'Oosterlinck.
This was based on previous work, [Word-Level Coreference Resolution](https://aclanthology.org/2021.emnlp-main.605/)
by Vladimir Dobrovolskii.

If you use the Stanza coref implementation in your work, please cite both of the following:

> Karel D'Oosterlinck, Semere Kiros Bitew, Brandon Papineau, Christopher Potts, Thomas Demeester, and Chris Develder. 2023. [CAW-coref: Conjunction-Aware Word-level Coreference Resolution.](https://arxiv.org/abs/2310.06165) In [CRAC 2023](https://sites.google.com/view/crac2023/). \[[pdf](https://arxiv.org/pdf/2310.06165.pdf)\]
{: .citation }

> Vladimir Dobrovolskii.  2021.  [Word-Level Coreference Resolution](https://aclanthology.org/2021.emnlp-main.605)  In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing.  \[[pdf](https://aclanthology.org/2021.emnlp-main.605.pdf)\]
{: .citation }

## Pipeline

Currently there is just one model available, an English model trained
on OntoNotes using Electra-Large.  Because this uses a transformer,
whereas the rest of the standard pipeline does not, this is not loaded
by default.  By adding it to the list of annotators, however, Stanza
will download the model and add it to the pipeline.

```python
import stanza
pipe = stanza.Pipeline("en", processors="tokenize,coref")
```

We found that a full finetuning meant the model files were quite large
relative to the rest of the pipeline, whereas not finetuning meant the
model was much less accurate.  However, using
[peft](https://github.com/huggingface/peft) produced an accurate model
with much less overhead.

The CAW-Coref paper uses Roberta-Large, and in fact we found that to
be slightly more accurate, but the rest of the transformer was using
Electra-Large on account of being slightly more accurate on other
tasks.  We may revisit that in the future.

Future work will include adding additional languages and building models which use less resources.

## API

When used in a pipeline, the coreference found will be attached at the `[Document](data_objects.md#document)` level.  The format is a sequence of `CorefChain` objects, where each `CorefChain` includes one or more `CorefMention` objects.  See [`coref_chain.py`](https://github.com/stanfordnlp/stanza/blob/main/stanza/models/coref/coref_chain.py) for more information.  The [`Word`](data_objects.md#word) objects each have a field `coref_chains` which is a list of `CorefAttachment` objects.  Those link back to the relevant chain and mark whether or not this is the representative mention.

The representative mention is chosen by taking the longest mention in a chain, ties broken by earliest in the document.

## Output

There are two output formats which include the coref results.

When coref is added, the `json` output format will include blocks such as the following:

```
>>> import stanza
>>> pipe = stanza.Pipeline("en", processors="tokenize,coref")
>>> pipe("John Bauer works at Stanford.  He has been there 4 years")
[
  [
    {
      "id": 1,
      "text": "John",
      "start_char": 0,
      "end_char": 4,
      "coref_chains": [
        {
          "index": 0,
          "representative_text": "John Bauer",
          "is_start": true,
          "is_representative": true
        }
      ]
    },
    {
      "id": 2,
      "text": "Bauer",
      "start_char": 5,
      "end_char": 10,
      "coref_chains": [
        {
          "index": 0,
          "representative_text": "John Bauer",
          "is_end": true,
          "is_representative": true
        }
      ]
    },
  ...
    {
      "id": 1,
      "text": "He",
      "start_char": 31,
      "end_char": 33,
      "coref_chains": [
        {
          "index": 0,
          "representative_text": "John Bauer",
          "is_start": true,
          "is_end": true
        }
      ]
    },
  ...
]
```

The conll format adds the annotations to the misc column:

```
>>> doc = pipe("John Bauer works at Stanford.  He has been there 4 years")
>>> print("{:C}".format(doc))
# text = John Bauer works at Stanford.
# sent_id = 0
1       John    _       _       _       _       0       _       _       start_char=0|end_char=4|coref_chains=start-repr-id0
2       Bauer   _       _       _       _       1       _       _       start_char=5|end_char=10|coref_chains=end-repr-id0
3       works   _       _       _       _       2       _       _       start_char=11|end_char=16
4       at      _       _       _       _       3       _       _       start_char=17|end_char=19
5       Stanford        _       _       _       _       4       _       _       start_char=20|end_char=28
6       .       _       _       _       _       5       _       _       start_char=28|end_char=29

# text = He has been there 4 years
# sent_id = 1
1       He      _       _       _       _       0       _       _       start_char=31|end_char=33|coref_chains=unit-id0
2       has     _       _       _       _       1       _       _       start_char=34|end_char=37
3       been    _       _       _       _       2       _       _       start_char=38|end_char=42
4       there   _       _       _       _       3       _       _       start_char=43|end_char=48
5       4       _       _       _       _       4       _       _       start_char=49|end_char=50
6       years   _       _       _       _       5       _       _       start_char=51|end_char=56
```

This is the first release of coref, so if there are suggested improvements to the format, please feel free to discuss this format [as a github issue](https://github.com/stanfordnlp/stanza/issues/new).

## Available Languages

Currently, there are models available for several languages, with more to come:

| Language | Dataset |
| :---     | :------ |
| CA       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |
| CS       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |
| DE       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |
| EN       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |
| ES       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |
| FR       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |
| HI       | DeepH   |
| NB       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |
| NN       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |
| PL       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |
| RU       | [CorefUD](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5478) |