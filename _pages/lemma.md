---
layout: page
title: Lemmatization
keywords: lemma, lemmatization, LemmaProcessor
permalink: '/lemma.html'
nav_order: 7
parent: Neural Pipeline
---

## Description

The lemmatization module recovers the lemma form for each input word. For example, the input sequence "I ate an apple" will be lemmatized into "I eat a apple". This type of word normalization is useful in many real-world applications. In Stanza, lemmatization is performed by the `LemmaProcessor` and can be invoked with the name `lemma`.

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| lemma | LemmaProcessor | tokenize, mwt, pos | Perform [lemmatization](https://en.wikipedia.org/wiki/Lemmatisation) on a [`Word`](data_objects.md#word) using the `Word.text` and `Word.upos` values. The result can be accessed as `Word.lemma`. | Generates the word lemmas for all words in the Document. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| lemma_use_identity | bool | `False` | When this flag is used, an identity lemmatizer (see `models.identity_lemmatizer`) will be used instead of a statistical lemmatizer. This is useful when [`Word.lemma`] is required for languages such as Vietnamese, where the lemma is identical to the original word form. |
| lemma_batch_size | int | 50 | When annotating, this argument specifies the maximum number of words to batch for efficient processing. |
| lemma_ensemble_dict | bool | `True` | If set to `True`, the lemmatizer will ensemble a seq2seq model with the output from a dictionary-based lemmatizer, which yields improvements on many languages (see system description paper for more details). |
| lemma_dict_only | bool | `False` | If set to `True`, only a dictionary-based lemmatizer will be used. For languages such as Chinese, a dictionary-based lemmatizer is enough. |
| lemma_edit | bool | `True` | If set to `True`, use an edit classifier alongside the seq2seq lemmatizer. The edit classifier will predict "shortcut" operations such as "identical" or "lowercase", to make the lemmatization of long sequences more stable. |
| lemma_beam_size | int | 1 | Control the beam size used during decoding in the seq2seq lemmatizer. |
| lemma_pretagged | bool | `False` | Assume the document is tokenized and pretagged. Only run lemma analysis on the document. |
| lemma_max_dec_len | int | 50 | Control the maximum decoding character length in the seq2seq lemmatizer. The decoder will stop if this length is achieved and the end-of-sequence character is still not seen. |
| lemma_store_results | bool | `True` | Remember the lemma predicted for each `(word, upos)` pair and reuse it the next time that pair appears, skipping the seq2seq model for words it has already seen. Words a contextual classifier decides, such as English `'s`, are never remembered, since their lemma depends on context. |
| lemma_store_results_size | int | 10000 | How many `(word, upos)` pairs `lemma_store_results` keeps. Once full, the least recently used pair is dropped, so memory stays bounded however long the pipeline runs. |

## Example Usage

Running the [LemmaProcessor](lemma.md) requires the [TokenizeProcessor](tokenize.md), [MWTProcessor](mwt.md), and [POSProcessor](pos.md).
After the pipeline is run, the [`Document`](data_objects.md#document) will contain a list of [`Sentence`](data_objects.md#sentence)s, and the [`Sentence`](data_objects.md#sentence)s will contain lists of [`Word`](data_objects.md#word)s.
The lemma information can be found in the `lemma` field of each [`Word`](data_objects.md#word).

### Accessing Lemma for Word

Here is an example of lemmatizing words in a sentence and accessing their lemmas afterwards:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'word: {word.text+" "}\tlemma: {word.lemma}' for sent in doc.sentences for word in sent.words], sep='\n')
```

As can be seen in the result, Stanza lemmatizes the word _was_ as _be_.

```
word: Barack    lemma: Barack
word: Obama     lemma: Obama
word: was       lemma: be
word: born      lemma: bear
word: in        lemma: in
word: Hawaii    lemma: Hawaii
word: .         lemma: .
```

### Lemmatizing pretagged text

If you already have tokenized, tagged text, you can use the lemmatizer to add lemmas without retokenizing or tagging:

```python
import stanza
from stanza.models.common.doc import Document

nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=True)
pp = Document([[{'id': 1, 'text': 'puppies', 'upos': 'NOUN'}]])
print("BEFORE ADDING LEMMA")
print(pp)
doc = nlp(pp)
print("AFTER ADDING LEMMA")
print(doc)
```

The updated doc will have the lemmas attached to the words:

```
BEFORE ADDING LEMMA
[
  [
    {
      "id": 1,
      "text": "puppies",
      "upos": "NOUN"
    }
  ]
]
AFTER ADDING LEMMA
[
  [
    {
      "id": 1,
      "text": "puppies",
      "lemma": "puppy",
      "upos": "NOUN"
    }
  ]
]
```


### Improving the Lemmatizer by Providing Key-Value Dictionary

It is possible to improve the lemmatizer by adding entries to its
dictionary.  The lemmatizer checks the dictionary first, and only uses
the statistical model for words which are not in the dictionary.

The dictionary is stored in the model file as `{upos: {word: lemma}}`,
compressed with gzip.  The lemmatizer first looks for the word under
its UPOS, then under the special key `"*"`, which holds entries for any
UPOS.  Words are matched exactly, including capitalization, unless the
model is caseless.

To add entries, load the model file, unpack the dictionary, add your
entries, and save the model under a new name:

```python
import gzip
import json
import os

import stanza
import torch

# find the lemmatizer model used by the default English pipeline
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')
model_file = nlp.processors['lemma'].config['model_path']
customized_file = os.path.join(os.path.dirname(model_file), 'customized_lemmatizer.pt')

model = torch.load(model_file, map_location='cpu', weights_only=True)
pos_dict = json.loads(gzip.decompress(model['dicts']))

# myword as a NOUN has the lemma mylemma
pos_dict.setdefault('NOUN', {})['myword'] = 'mylemma'
# myword with any other UPOS has the lemma mylemma as well
pos_dict.setdefault('*', {})['myword'] = 'mylemma'

model['dicts'] = gzip.compress(json.dumps(pos_dict).encode('utf-8'))
torch.save(model, customized_file)

# load the customized model with Stanza
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', lemma_model_path=customized_file)
print(nlp('myword'))  # should get lemma 'mylemma'
```

As can be seen in the result, Stanza should lemmatize the word _myword_ as _mylemma_.

This dictionary format is used starting with Stanza 1.15.  In 1.13 and
earlier, `model['dicts']` was a tuple of two dictionaries, one mapping
`(word, upos)` to `lemma` and one mapping `word` to `lemma`.  Models
saved by 1.14 used a pickled version of the current format, which is
no longer loaded for security reasons.  Models from either version can
be converted to the current format with
[stanza/utils/lemma/convert_lemma_dict.py](https://github.com/stanfordnlp/stanza/blob/main/stanza/utils/lemma/convert_lemma_dict.py).

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/main/stanza/models/lemmatizer.py#L22) of the lemmatizer.

### Training with an external dictionary

**Coming soon:** the scripts and options in this section are on the
[`dev` branch](https://github.com/stanfordnlp/stanza/tree/dev), and
will be part of the next code release.

A lemmatizer can be trained with an external lexicon as well as a
treebank.  The default English, German, and Slovenian models do this,
with the SPECIALIST Lexicon, German Wiktionary, and Sloleks.  See
[Combined models](combined_models.md#lemma-dictionaries) for where
those lexicons come from and the terms of their licenses.

The lexicon is used in two ways:

- Every entry goes into the lemmatizer's dictionary.  Where the lexicon and the treebank disagree on the lemma for the same word and UPOS, the treebank's lemma wins.
- The seq2seq model also trains on the lexicon, mixed in with the treebank batches.  This teaches it inflection patterns which the treebank has too few examples of, so it does better on words which are in neither one.

The lexicon is converted when the dataset is prepared.  For
`de_combined`, `en_combined`, and `sl_combined`, `prepare_lemma_treebank` reads the
lexicon from `extern_data` and writes it next to the train, dev, and
test files as `{short_name}.dict.conllu`:

| Dataset | Lexicon | Expected location |
| :------ | :------ | :---------------- |
| en_combined | [`inflVars.data`](https://data.lhncbc.nlm.nih.gov/public/lsg/lexicon/2026/release/LEX_DOC/MISC/inflVars.data) from the [2026 SPECIALIST Lexicon release](https://lhncbc.nlm.nih.gov/LSG/Projects/lexicon/current/web/release/2026.html) | `extern_data/english/specialist/inflVars.data` |
| de_combined | `de_wiki_lemmas.conllu.gz`, already converted from German Wiktionary | `extern_data/handparsed-treebank/german-lemmas-wiki/de_wiki_lemmas.conllu.gz` |
| sl_combined | [Sloleks 3.1](http://hdl.handle.net/11356/2080) XML files | `extern_data/slovenian/Sloleks.3.1/` |

```bash
python3 -m stanza.utils.datasets.prepare_lemma_treebank en_combined
```

If the lexicon is missing, preparing the dataset stops with an error.
To prepare it without the lexicon, use `--no_external_dictionary`,
which also removes a dictionary file left from an earlier run.

`run_lemma` then uses `data/lemma/{short_name}.dict.conllu`
automatically if it exists:

```bash
python3 -m stanza.utils.training.run_lemma en_combined
```

| Option | Script | Description |
| --- | --- | --- |
| `--no_external_dictionary` | `prepare_lemma_treebank` | Don't write the lexicon for a dataset which has one, and remove an existing `{short_name}.dict.conllu`. |
| `--secondary_file` | `run_lemma` | Use this file as the external dictionary instead of the prepared one.  This works for any dataset, not just the ones with a default lexicon. |
| `--no_use_secondary_file` | `run_lemma` | Train without an external dictionary, even if one was prepared. |
| `--secondary_rate` | `run_lemma`, `lemmatizer` | How many batches of the lexicon the seq2seq model trains on for each batch of the treebank.  The default of 0.1 means one lexicon batch after every 10th treebank batch.  Batches are counted in words, and the lexicon is reshuffled after each pass through it.  0 uses the lexicon only for the dictionary. |

To use a lexicon of your own, write it as CoNLL-U with one word per
sentence, filling in the form, lemma, and UPOS:

```
# sent_id = my_lexicon-NOUN-housewives
# text = housewives
1	housewives	housewife	NOUN	_	_	0	root	_	_
```

and pass it to `run_lemma` with `--secondary_file`.

## Contextual Classifier

As of Stanza 1.10, there is a contextual classifier for words with
ambiguous words that can be determined by context.  For example, in
English, the token `'s` as a verb can either represent `is` or `has`,
with the lemma `be` or `have` respectively.

One caveat is that this requires at least a few examples to have
reasonable results.  Therefore, although there are other possible
examples in English such as `saw`, `wound`, and `found`, the existing
training data does not cover those ambiguous lemmas.

Note that in general POS is sufficient to distinguish many possible
ambiguous lemmas, and the contextual lemmatizer is only needed for
cases where the POS (specifically the UPOS) are the same for two
possible resolutions.  For example, in English, the ambiguity of `'s`
as a verb or as a possessive was never an issue.

Candidates for this expansion based on exploring UD treebanks can be
found in
[stanza/utils/datasets/prepare_lemma_classifier.py](https://github.com/stanfordnlp/stanza/blob/main/stanza/utils/datasets/prepare_lemma_classifier.py).
Currently, the released models only use this for `'s` in English and
`के` in Hindi, although we can expand this later, and more suggestions
or data for ambiguous cases with little available data are welcome.
