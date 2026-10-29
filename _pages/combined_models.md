---
layout: page
title: Combined models
keywords: combined models
permalink: '/combined_models.html'
nav_order: 5
parent: Models
---

## Combined models

The default models for some languages are "combined".  The goal is to get better coverage of the language, hopefully without sacrificing consistency in the annotation scheme.  In each case, the data used to train the models is a combination of multiple UD datasets.

| Language | UD Datasets | Other |
| :------- | :------------ | :------ |
| English | EWT, GUM, GUMReddit, PUD, Pronouns | [SPECIALIST Lexicon](#specialist-lexicon) lemma dictionary (coming soon) |
| French  | GSD, ParisStories, Rhapsodie, Sequoia | |
| German  | GSD | [Lemmas data composed from German Wiktionary](https://github.com/stanfordnlp/handparsed-treebank/blob/master/german-lemmas-wiki/de_wiki_lemmas.conllu) |
| Hebrew  | IAHLTwiki | [HTB fork from IAHLT](https://github.com/IAHLT/UD_Hebrew) |
| Italian | ISDT, VIT, PoSTWITA, and TWITTIRO | [MWT list from Prof. Attardi](https://github.com/stanfordnlp/handparsed-treebank/blob/master/italian-mwt/italian.mwt) |
| Japanese | GSD | [Parsed data from CMU](https://github.com/stanfordnlp/handparsed-treebank/blob/master/japanese-handparsed/spaces-ready-checked.conllu) |
| Slovenian | SSJ, SST | [SUK](https://www.clarin.si/repository/xmlui/handle/11356/1959) ssj500k and ambiga for POS and lemmas, [Sloleks](#sloleks) lemma dictionary |
| Spanish | AnCora, GSD | [MWT and lemmas for clitics and adjectives](https://github.com/stanfordnlp/handparsed-treebank/blob/master/spanish-mwt/adjectives.conllu) |

Other data sets would be added, or combined models for other languages
created, but there are often data problems preventing that.  For
example, English Lines uses a different xpos and feature scheme.
Spanish GSD and AnCora use the same general annotation scheme, but
sentence splits are annotated differently, and some of the features
are noticeably different.  Hopefully over time we can resolve some of
those issues and expand the models.

Whether or not this was a good idea was explored [in a GURT paper from Georgetown](https://arxiv.org/abs/2302.00636)

## Lemma dictionaries

The Slovenian lemmatizer, and starting with the next model release
the English lemmatizer, also use a large external lexicon, to cover
words which never appear in the treebanks.  Each lexicon is converted
to a list of `(word, UPOS, lemma)` entries, and those entries go into
the lemmatizer's dictionary.  Where a treebank and a lexicon disagree
on the lemma for the same word and UPOS, the treebank's lemma is the
one in the dictionary.

Starting with the next model release, the seq2seq model also trains
on the lexicon, at a lower rate than the treebank data, so that it
learns the inflection patterns as well as the specific words.

See [the lemmatizer page](lemma.md#training-with-an-external-dictionary)
for how to rebuild these dictionaries when training a model.

### SPECIALIST Lexicon

**Coming soon:** the English models with this dictionary will be part
of the next model release.

The English lemma dictionary is built from the inflection table
([`inflVars.data`](https://data.lhncbc.nlm.nih.gov/public/lsg/lexicon/2026/release/LEX_DOC/MISC/inflVars.data))
of the [2026 release](https://lhncbc.nlm.nih.gov/LSG/Projects/lexicon/current/web/release/2026.html)
of the SPECIALIST Lexicon, part of the
[SPECIALIST NLP Tools](https://lhncbc.nlm.nih.gov/LSG/Projects/lexicon/current/web/index.html)
from the Lister Hill National Center for Biomedical Communications,
National Library of Medicine.  The SPECIALIST Lexicon is a United
States Government product, and its use and redistribution are subject
to the following conditions, from the
[SPECIALIST NLP Tools Terms and Conditions](https://lhncbc.nlm.nih.gov/LSG/Docs/termsAndConditions.html):

- Redistributions of the SPECIALIST NLP Tools in source or binary form must include this list of conditions in the documentation and/or other materials provided with the distribution.
- In any publication or distribution of all or any portion of the SPECIALIST NLP Tools (1) you must attribute the source of the tools as the SPECIALIST NLP Tools with the release number and date; (2) you must state any modifications made to the SPECIALIST NLP Tools along with a complete description of the modifications, which may be in the form of patch files.
- You shall not assert any proprietary rights to any portion of the SPECIALIST NLP Tools, nor represent the SPECIALIST NLP Tools or any part thereof to anyone as other than a United States Government product.
- The name of the U.S. Department of Health and Human Services, National Institutes of Health, National Library of Medicine, Lister Hill National Center for Biomedical Communications may not be used to endorse or promote products derived from the SPECIALIST NLP Tools without specific prior written permission.
- Neither the United States Government, U.S. Department of Health and Human Services, National Institutes of Health, National Library of Medicine, Lister Hill National Center for Biomedical Communications, nor any of its agencies, contractors, subcontractors or employees of the United States Government make any warranties, expressed or implied, with respect to the SPECIALIST NLP Tools, and, furthermore, assume no liability for any party's use, or the results of such use, of any part of these tools.

The modifications made to the lexicon are:

- Only the noun, verb, adjective, and adverb rows are kept, mapped to the UPOS tags `NOUN`, `VERB`, `ADJ`, and `ADV`.  The lexicon has no category for proper nouns, so a noun whose lemma has a capital letter, such as `Alzheimer`, is tagged `PROPN`.
- The lemma is the uninflected form of each row, which keeps the spelling of the inflected form (`colours` → `colour`), rather than the citation form.
- Rows are dropped if the form or lemma contains whitespace, if the form is a negated contraction such as `didn't`, or if the form or lemma has characters other than letters, digits, `-`, `'`, and `.`.
- A `(word, UPOS)` pair with more than one lemma, such as `axes` → `axe` / `axis`, is dropped.
- The remaining entries are written as one-word CoNLL-U sentences.

The complete conversion is
[stanza/utils/datasets/lemma/convert_en_specialist.py](https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/datasets/lemma/convert_en_specialist.py).

### Sloleks

The Slovenian lemma dictionary is built from
[Sloleks 3.1](http://hdl.handle.net/11356/2080) (Čibej, Gantar,
Dobrovoljc, Krek, Holozan, Erjavec, Romih, Arhar Holdt, Krsnik, and
Robnik-Šikonja, 2026, Slovenian language resource repository
CLARIN.SI), which is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The lexicon is converted to `(word, UPOS, lemma)` entries, keeping the
orthographic forms of each headword.  Conjunctions, abbreviations, and
residuals are skipped, and the other categories are mapped to UPOS.
The conversion is
[stanza/utils/datasets/lemma/convert_sl_sloleks.py](https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/datasets/lemma/convert_sl_sloleks.py).

## Data augmentation

In general, the models also use various sorts of text "data
augmentation" of the original data.

For example, in a language where all of the sentences in the POS
training data end with punctuation, we remove the trailing punctuation
from some fragment of sentences to train the model to handle sentences
which don't end with sentence final punctuation.  Otherwise, we would
frequently get issues such as a model tagging `This is an unfinished
sentence_PU`

The tokenization models also use a couple different versions of this.
For example, most of the tokenization datasets have just one or a
couple forms of quotes, but we replace some fraction of quotes with
different types so that each model has a chance of correctly
tokenizing `«data augmentation»`, `"data augmentation"`, etc.  Also,
sentence final punctuation will often have spaces added or removed to
make the model more robust to typos.
