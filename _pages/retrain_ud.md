---
layout: page
title: Retrain models for a UD dataset
keywords: stanza, model training
permalink: '/retrain_ud.html'
nav_order: 9
parent: Usage
---

## Retrain UD models

There could be several reasons to retrain models for an existing UD dataset.  For example:

- code changes which affect the quality of the model or the compatibility with existing serialized models
- new version of UD released
- updated word vectors
- proposed changes to UD dataset available in git or elsewhere

Here we will present an end to end example on how to retrain the
models based on UD: tokenizer, MWT, lemmatizer, POS, depparse.

The below guide uses some changes to the codebase which will be available in Stanza 1.4.1.  In the meantime, you can git clone the dev branch of our repo:

https://github.com/stanfordnlp/stanza

### OS

We will work on Linux for this.  It is possible to recreate most of
these steps on Windows or another OS, with the exception that the
environment variables need to be set differently.

As a reminder, Stanza only supports Python 3.6 or later.
If you are using an earlier version of Python, it will not work.

### Environment variables

To start, we will retrain the tokenizer module for UD_English-EWT.
The default environment variables work well (especially if you're me!)
but may not be applicable to your system.  

To rebuild the tokenizer, there are a few relevant environment variables:

- `PYTHONPATH` - if using a git clone of Stanza, you will want to set
  your `PYTHONPATH` to the home directory of that checkout.  You can
  also use `.` if you are running the scripts from that directory
- `UDBASE` - the home path of a Universal Dependencies download.  For
  example, if you have a complete download of 2.10, you can set it to
  that directory:
  `UDBASE=/u/scr/corpora/Universal_Dependencies/Universal_Dependencies_2.10/ud-treebanks-v2.10/`
  You will have this set correctly if `ls $UDBASE` displays all of the
  UD packages, such as `UD_Afrikaans-AfriBooms`, `UD_Yupik-SLI`, and
  many more in between.  You can also set up a specific directory for
  git checkouts if you want to track the latest changes, such as
  `UDBASE=/home/$USER/ud/git`
- `TOKENIZE_DATA_DIR` - by default, Stanza will write preprocessed datasets to
  `$DATA_ROOT/tokenize`, which defaults to `data/tokenize`
- `STANZA_RESOURCES_DIR` - Supplemental models such as default word
  vectors will be downloaded here (although the tokenizer in
  particular does not use word vectors).  You can change where they go
  by changing this variable.

This might look like this in a `.bashrc` file:

```bash
export PYTHONPATH=.

export STANZA_RESOURCES_DIR=/nlp/scr/$USER/stanza_resources
export TOKENIZE_DATA_DIR=/nlp/scr/$USER/data/tokenize
# if using a complete installation of UD
export UDBASE=/u/scr/corpora/Universal_Dependencies/Universal_Dependencies_2.10/ud-treebanks-v2.10/
# if using a personal git install of some of the datasets
export UDBASE=/nlp/scr/$USER/Universal_Dependencies/git
```

### Obtaining data

All of the UD based models use data available at [Universal Dependencies](https://universaldependencies.org/)

Individual language/dataset pairs are each in their own github repo.
For this tutorial, we will use the UD English EWT repo.  In July 2022,
for example, this repo was updated with morphological feature changes
which would not be released until UD 2.11 in November 2022.

The first step is to git clone [UD_English-EWT](https://github.com/UniversalDependencies/UD_English-EWT):

```bash
cd $UDBASE
git clone git@github.com:UniversalDependencies/UD_English-EWT.git
cd UD_English-EWT
git checkout dev    # because we want the dev set updates, of course
```

{% include alerts.html %}
{{ note }}
{{ "If you are using the latest official release, you do not need to clone a UD repo" | markdownify }}
{{ end }}

Instead of downloading an individual repo, we can download them all
[from the UD home page](https://universaldependencies.org/#download)
and put that in `$UDBASE`.

### Preparing data

A script is included with Stanza which will read the dataset from `$UDBASE` and write it to `$TOKENIZE_DATA_DIR`:

```bash
python3 -m stanza.utils.datasets.prepare_tokenizer_treebank UD_English-EWT
```

### Running the model

There is also a script to run the model

```bash
python3 -m stanza.utils.training.run_tokenizer UD_English-EWT
```

This will train the model and put the result in `saved_models/tokenize/en_ewt_tokenizer.pt`

You can change the destination directory with `--save_dir`:

```bash
python3 -m stanza.utils.training.run_tokenizer UD_English-EWT --save_dir somewhere/else
```

If you already have a model saved, the training script will not overwrite that model.  You can make that happen with `--force`:

```bash
python3 -m stanza.utils.training.run_tokenizer UD_English-EWT --force
```

If you want a different save name:

```bash
python3 -m stanza.utils.training.run_tokenizer UD_English-EWT --save_name en_ewt_variant_tokenizer.pt
```

### Testing the model

After a model is built, you can test it on the dev and test sets:

```bash
python3 -m stanza.utils.training.run_tokenizer UD_English-EWT --score_dev
python3 -m stanza.utils.training.run_tokenizer UD_English-EWT --score_test
```

### Other processors

The process is identical for MWT, Lemmatizer, and POS.

There is a subtle difference for Depparse, though.  TODO

Note that MWT does not apply for datasets with no multi-word tokens.  If you attempt to run MWT on a dataset with no MWTs, you will get a message such as

```
2022-08-01 18:11:03 INFO: No training MWTS found for vi_vtb.  Skipping
```

| Model    | Prepare script                                   | Run script                         | Data dir env variable | Default save dir      |
| :----    | :----                                            | :-----                             | :-------              | :----                 |
| MWT      | stanza.utils.datasets.prepare_mwt_treebank       | stanza.utils.training.run_mwt      | MWT_DATA_DIR          | saved_models/mwt      |
| Lemma    | stanza.utils.datasets.prepare_lemma_treebank     | stanza.utils.training.run_lemma    | LEMMA_DATA_DIR        | saved_models/lemma    |
| POS      | stanza.utils.datasets.prepare_pos_treebank       | stanza.utils.training.run_pos      | POS_DATA_DIR          | saved_models/pos      |
| Depparse | stanza.utils.datasets.prepare_depparse_treebank  | stanza.utils.training.run_depparse | DEPPARSE_DATA_DIR     | saved_models/depparse |

