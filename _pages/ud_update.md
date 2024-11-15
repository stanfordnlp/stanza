---
layout: page
title: Retrain models for a new UD release
keywords: stanza, model training
permalink: '/update_ud.html'
nav_order: 10
parent: Training
---


steps for updating to a new version of UD:

### Build reverse treebank name map

```
python3 stanza/models/common/build_short_name_to_treebank.py
```

there may be some errors such as
```
ValueError: Unable to find language code for Maghrebi_Arabic_French
```

If so, find the language code which was used in UD by looking in the appropriate directory.
In the case of UD 2.12, `ud-treebanks-v2.12/UD_Maghrebi_Arabic_French-Arabizi`
Add that to the languages in `stanza/models/common/constant.py`

### Possibly update Chinese datasets

Weird edge case that occasionally happens: new ZH dataset, check if it is simplified or traditional.  The `build_short_name_to_treebank.py` script will throw an exception if this happens.

Add the appropriate line to the special cases in `constant.py`.
TODO: could add a script which checks for this

### Rebuild xpos vocab map

Although not strictly necessary, there is a script which
precalculates the xpos vocabs for each of the known datasets.  You can
rerun it by doing TODO to check that the derived xpos vocabs all make
sense.

### Prepare datasets and run the training scripts

There are 5 annotators derived directly from UD data: tokenizer, MWT, POS, lemma, depparse.  For each of those, do the following:

```
python3 stanza/utils/datasets/prepare_annotator_treebank.py ud_all
python3 stanza/utils/training/run_annotator.py ud_all
```

Note that while these don't exactly need to be done separately, and
much of this work can be done simultaneously, there is one specific
ordering which must be followed.  The dependency parsers use predicted
POS tags for the dependencies, so the POS must be done before the
depparse.

### Find new word vectors

If you get an error akin to the following, you will need to find and add word vectors for the new language to Stanza.

```
FileNotFoundError: Cannot find any pretrains in /home/john/stanza_resources/qaf/pretrain/*.pt  No pretrains in the system for this language.  Please prepare an embedding as a .pt and use --wordvec_pretrain_file to specify a .pt file to use
```

You can do this by putting new word vectors in the collection of
models to be rebuilt, in `<lang>/pretrain`, then rebuild the models
distribution with `prepare_resources.py`.  See below for more details.

If you can't find a reasonable embedding for a language, such as Old
East Slavic, you can add that to the `no_pretrain_languages` set which
currently resides in `stanza/resources/default_packages.py`
(although there is no guarantee that map doesn't move somewhere else)

### Add new default packages

Any new languages will also need a default package marked in the appropriate resources file, probably
`stanza/resources/default_packages.py`

### Tokenizer models with dictionaries

Some of the tokenizer models have dictionaries.  To get the best
results, you will want to reuse the dictionaries, possibly by
extracting them from the current models.

### Combined models

There are some "combined" models which are composed of multiple UD
datasets and/or external resources.  At a minimum, those should be EN,
IT, HE, and FR, but there may be others by the time someone is trying
to follow this script.  Rerun the training scripts for those models.

For example

```
python3 stanza/utils/datasets/prepare_tokenizer_treebank.py en_combined
python3 stanza/utils/training/run_tokenizer.py en_combined
```


### Rebuild with charlm & bert

Currently the mechanism for rerunning all of the models w/ and w/o
charlm, w/ and w/o transformers is kind of janky.  You will need to
pay special attention to that.  Future versions will hopefully have a
better integration of that.  In particular, this applies to
lemmatizer, POS, and depparse for now.  Sorry for that.

### Gather all of the models

Gather all of the new models in a central location.
On our system, we have been using
`/u/nlp/software/stanza/models/`
so models wind up in
`/u/nlp/software/stanza/models/current-models-1.5.1/pretrain`,
`/u/nlp/software/stanza/models/current-models-1.5.1/tokenize`, etc

If you copy the models to a new directory, use `cp -r --preserve` so
that the metadata on the original models is kept.  The above process
will not update tokenizers built from non-UD resources, such as the
`sd_isra` tokenizer, or constituency parsers, NER models, etc.  Not
updating the metadata will save quite a bit of time & bandwidth when
reuploading those to the huggingface git repos.

If starting from scratch, don't forget to collect all of the other
models - tokenizers built from non-UD sources, pretrains, charlms,
etc.  Again, use `cp --preserve` to keep the file metadata the same.

### Build a model distribution

Use the `stanza/resources/prepare_resources.py` script to rebuild
those models into a new distribution package

### Update the resources.json file

Update the `resources.json` file for the appropriate distribution.
On the Stanford NLP file system, that is in
`/u/downloads/software/stanza/stanza-resources`

### Push models

Push all of the models to HuggingFace:
```
cd /u/nlp/software/
python3 huggingface-models/hugging_stanza.py
```
