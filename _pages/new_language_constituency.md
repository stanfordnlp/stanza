---
layout: page
title: Adding a new Constituency model
keywords: constituency, stanza, model training
permalink: '/new_language_constituency.html'
nav_order: 7
parent: Training
---

## End to End Constituency Example

I refuse to believe anyone actually wants to do this.

Nevertheless, it is actually possible to do.  We will follow along with the VIT dataset from ELRA:

http://catalog.elra.info/en-us/repository/browse/ELRA-W0040/

At each step of the way, please mentally substitute the language and dataset of your choice for Italian VIT.

Some of the features (such as silver trees) mentioned below are only
available in the coming version, 1.5.0.  Since you will be using a
clone of our git repo to do this, such distinctions don't matter.

### OS

We will work on Linux for this.  It is possible to recreate most of
these steps on Windows or another OS, with the exception that the
environment variables need to be set differently.

As a reminder, Stanza only supports Python 3.6 or later.
If you are using an earlier version of Python, it will not work.

### Codebase

This is a previously unknown dataset, so it will require some code
changes.  Accordingly, we first clone
[the stanza git repo](https://github.com/stanfordnlp/stanza)
and check out the `dev` branch.  We will then create a new branch
for our code changes.

```bash
git clone git@github.com:stanfordnlp/stanza.git
cd stanza
git checkout dev
git checkout -b italian_vit
```

### Environment

There are two environment variables which are relevant for building
constituency models.  They are `$CONSTITUENCY_BASE`, which represents
where the *raw*, unconverted constituency data lives, and
`$CONSTITUENCY_DATA_DIR`, which represents where the processed data
will go.


Both of these have reasonable defaults, but we can still customize them.

`$CONSTITUENCY_BASE` determines where the *raw, unchanged* datasets go.

The purpose of the data preparation scripts will be to put *processed*
forms of this data in `$CONSTITUENCY_DATA_DIR`.  Once this is done, the
execution script will expect to find the data in that directory.


In `~/.bashrc`, we can add the following lines.  Here are a couple
values we use on our cluster to organize shared data:

{% include alerts.html %}
{{ note }}
{{ "Your OS may use a different startup script than `~/.bashrc`" | markdownify }}
{{ end }}

{% include alerts.html %}
{{ note }}
{{ "On Windows, you can update these variables in `Edit the system environment variables` in the Control Panel" | markdownify }}
{{ end }}

```bash
export CONSTITUENCY_BASE=/u/nlp/data/constituency-parser
export CONSTITUENCY_DATA_DIR=/nlp/scr/$USER/data/constituency
```

Since you will be running python programs directly from the git
checkout of Stanza, you will need to make sure `.` is in your
`PYTHONPATH`.

```bash
user@host:...$ echo $PYTHONPATH
.
```

### Language code

If your dataset involves a language currently not in Stanza, it may
need [the language code](new_language.md#language-codes) added to
Stanza.  Feel free to open an issue on our github or send us a PR.

### Data download

In general, there is no standardization among constituency datasets
for format or layout.  In terms of where to store the data, so far we
have followed the convention of putting the raw data in
`$CONSTITUENCY_BASE/<lang>/<dataset>`.  So, using the
`$CONSTITUENCY_BASE` mentioned above, on our file systems, the `IT_VIT`
dataset can be found at `/u/nlp/data/constituency-parser/italian/it_vit`

Although it is not required to follow this convention, it will
certainly make things easier if you want to integrate your work with
ours, such as with a git pull request.

### Processing raw data to trees

Interally, the parser reads data in the format of PTB bracketed trees:

```
(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN sample)))))
```

Rather than making the tree reader process many different formats,
instead we convert the raw data files to PTB bracketed trees.

In the case of `it_vit`, for example, the raw input trees are in a
format that looks like:

```
ID#sent_00001 fc-[f3-[sn-[art-le, n-infrastrutture, sc-[ccom-come, sn-[n-fattore, spd-[pd-di, sn-[n-competitività]]]]]], f3-[spd-[pd-di, sn-[mw-Angela, nh-Airoldi]]], punto-.]
```

There are a couple things to note:

- Instead of `()`, it uses `[]` as brackets
- Instead of `(NN word)`, it uses `n-infrastrutture`
- Not shown in this sample: some phrases become single tokens, such as `n-polo_di_attrazione`
- Capitalization was discarded, but [the UD conversion of this treebank](https://github.com/UniversalDependencies/UD_Italian-VIT) has the capitalization restored

To be clear, each of these design decisions can be justified, but the
point is that we want to standardize the input format.  We therefore
wrote a conversion script
[specific for `it_vit`](https://github.com/stanfordnlp/stanza/blob/e431d69e4b12d564de545b814eb969cd34a1eea6/stanza/utils/datasets/constituency/convert_it_vit.py)
which reads the tree format used in the original VIT dataset, combines
the text with updated text in the UD conversion of VIT, and outputs
all of this as bracketed trees.  After running the script, the first
line of the data file becomes:

```
(ROOT (fc (f3 (sn (art Le) (n infrastrutture) (sc (ccom come) (sn (n fattore) (spd (pd di) (sn (n competitività))))))) (f3 (spd (pd di) (sn (mw Angela) (nh Airoldi)))) (punto .)))
```

Again, to emphasize, it is not necessary to follow along exactly with the conversion script.  The intent is that the final output will be PTB formatted trees:

- Open and close `()` denote subtrees
- The first entry in a subtree is the label on that subtree.  In `it_vit`'s first sentence, `ROOT` is the label of the entire tree, `fc` is the label of the first bracket, etc
- Leaves represent the words of the tree.  Reading from the start to the end of a tree reads off the words of the sentence in order
- POS tags are represented as "preterminals": eg, a bracket of exactly one word, with the label on the tree being the POS tag.  `(art Le)` is the preterminal for the first word of the first sentence in `it_vit`

The intent will be to write a new script which converts the specific dataset you are using to PTB style trees.

#### Conversion wrapper script

{% include alerts.html %}
{{ note }}
{{ "In order to accommodate Vietnamese, or possibly languages with spaces in the words, the PTB tree reader will combine multiple words at the bottom layer into one leaf.  So, for example, `(N-H tình hình)` becomes one leaf with the word "tình hình" | markdownify }}
{{ end }}

In addition to the conversion script, there is a
[wrapper script](https://github.com/stanfordnlp/stanza/blob/e431d69e4b12d564de545b814eb969cd34a1eea6/stanza/utils/datasets/constituency/prepare_con_dataset.py)
which calls the dataset-specific preparation script.  The intent is to organize all of the preparation scripts into one tool, so that one can run

```
python3 -m stanza.utils.datasets.constituency.prepare_con_dataset it_vit
python3 -m stanza.utils.datasets.constituency.prepare_con_dataset vi_vlsp22
etc etc
```

If you plan on contributing your conversion back to Stanza as a PR,
please add a function to `prepare_con_dataset.py` which calls the new
conversion script.

### POS tagging

There is an important consideration for the constituency parser.  The
parser does not have a POS tagger as part of the model.  Instead, it
relies on tags supplied to it earlier in the pipeline at both training
and test time.

Practically, this means that in order to have a usable model, there
*must* be a POS tagger for the language in question.  Otherwise, there
will be no tags available at parse time.

While technically it is possible for the parser to operate without
tags, currently that is not implemented as a feature.  It is also not
planned as a feature any time soon, although a PR which adds it would
be welcome.

When training (see below), the training script will attempt to use the
default POS package from Stanza for the given language.  To change to
a different POS package, one can use the `--retag_package` command
line flag.


### Word Vectors

The base version of the tool uses word vectors as an input layer to the classifier.

Please refer to the word vectors section of [NER models](new_language_ner.md#word-vectors) to add a new set of word vectors.

{% include alerts.html %}
{{ note }}
{{ "Stanza already has Italian word vectors, so we do not need to do anything to add them.  In fact, the `run_constituency` script will attempt to automatically download the default word vectors." | markdownify }}
{{ end }}


### Charlm and Bert

There is a pretrained character model that ships with Stanza, and
there is also support in the Constituency model for HuggingFace
transformers.  Both given substantial improvements in performance.
There is more description of how to use them in the corresponding
section of the [NER models](new_language_ner.md#charlm-and-bert) page.

{% include alerts.html %}
{{ note }}
{{ "Stanza also has a pretrained Italian charlm, so we do not need to do anything to add that.  The `run_constituency` script will also attempt to automatically download that model." | markdownify }}
{{ end }}


### Training!

At this point, everything is ready to push the button and start training.

```bash
python -m stanza.utils.training.run_constituency it_vit
```

### Running the dev and test sets

# TODO: add a section on this

### Useful flags

# TODO: add a section on this

### Silver Trees

# TODO: add a section on this

