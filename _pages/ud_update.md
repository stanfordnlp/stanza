---
layout: page
title: Retrain models for a new UD release
keywords: stanza, model training
permalink: '/update_ud.html'
nav_order: 8
parent: Training
---


steps for updating to a new version of UD:

1) rerun python3 stanza/models/common/build_short_name_to_treebank.py

there may be some errors such as
ValueError: Unable to find language code for Maghrebi_Arabic_French

If so, find the language code which was used in UD by looking in the appropriate directory
in this case, ud-treebanks-v2.12/UD_Maghrebi_Arabic_French-Arabizi
then add that to the languages in stanza/models/common/constant.py

2) weird edge case that occasionally happens: new ZH dataset, check if it is simplified or traditional.
add the appropriate line to the special cases in constant.py
TODO: could add a script which checks for this

3) Although not strictly necessary, there is a script which
precalculates the xpos vocabs for each of the known datasets.  You can
rerun it by doing TODO to check that the derived xpos vocabs all make
sense.

4) There are 5 annotators derived directly from UD data: tokenizer, MWT, POS, lemma, depparse.  For each of those, do the following:
```
python3 stanza/utils/datasets/prepare_annotator_treebank.py ud_all
python3 stanza/utils/training/run_annotator.py ud_all
```

Note that while these don't exactly need to be done separately, and
much of this work can be done simultaneously, there is one specific
ordering which must be followed.  The dependency parsers use predicted
POS tags for the dependencies, so the POS must be done before the
depparse.

5) If you get an error akin to the following, you will need to find and add word vectors for the new language to Stanza.

```
FileNotFoundError: Cannot find any pretrains in /home/john/stanza_resources/qaf/pretrain/*.pt  No pretrains in the system for this language.  Please prepare an embedding as a .pt and use --wordvec_pretrain_file to specify a .pt file to use
```

You can do this by putting new word vectors in the collection of
models to be rebuilt, in <lang>/pretrain, then rebuilt the models
distribution with prepare_resources.py.  See below for more details.

If you can't find a reasonable embedding for a language, such as Old
East Slavic, you can add that to the `no_pretrain_languages` set which
currently resides in `stanza/resources/default_packages.py`, although
there is no guarantee it doesn't move somewhere else

6) There are some "combined" models which are composed of multiple UD
datasets and/or external resources.  At a minimum, those should be EN,
IT, HE, and FR, but there may be others by the time someone is trying
to follow this script.  Repeat step 4 for those models.

7) Currently the mechanism for rerunning all of the models w/ and w/o
charlm, w/ and w/o transformers is kind of janky.  You will need to
pay special attention to that.  Future versions will hopefully have a
better integration of that.  In particular, this applies to
lemmatizer, POS, and depparse for now.

8) Gather all of the new models in a central location.  On our system, we have been using
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

9) Use the `stanza/resources/prepare_resources.py` script to rebuild
those models into a new distribution package

10) Update the `resources.json` file for the appropriate distribution.
On the Stanford NLP file system, that is in
`/u/downloads/software/stanza/stanza-resources`

11) Push all of the models to HuggingFace:
```
cd /u/nlp/software/
python3 huggingface-models/hugging_stanza.py
```
