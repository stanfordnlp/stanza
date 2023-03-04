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

Nevertheless, it is actually possible to do.  We will follow along with [the VIT dataset from ELRA](http://catalog.elra.info/en-us/repository/browse/ELRA-W0040/)

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

Sometimes, a language will not have a suitable POS tagger available.
In the case of a Vietnamese constituency parser we built, we found it
advantageous to use the constituency treebank for training a POS tagger,
as that dataset was much larger than the available UD dataset.
The script we used for converting the treebank to a POS dataset
can be used on any language:

[convert_trees_to_pos.py](https://github.com/stanfordnlp/stanza/blob/6b9ecae54bbb5b95d42c9732675180e3aa4653d3/stanza/utils/datasets/pos/convert_trees_to_pos.py)

This needs to be run on a fully converted treebank.  So, for example, one would run

```
python3 -m stanza.utils.datasets.constituency.prepare_con_dataset vi_vlsp22
python3 -m stanza.utils.datasets.pos.convert_trees_to_pos vi_vlsp22
python3 -m stanza.utils.training.run_pos vi_vlsp22
```

The same arguments for word vectors, charlm, and transformers apply to both `run_constituency` and `run_pos`

### CoreNLP installation

The parser uses a scoring script from CoreNLP to find the bracket F1 score.
Therefore, you need to follow the
[CoreNLP setup instructions](client_setup.md)
to download the software.
It is not necessary to download any of the CoreNLP models, though.

### Word vectors

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

The model will take quite some time to train, and you almost certainly want to use a GPU to do it.

### Running the dev and test sets

Once the model is trained, you can test it on the dev and test sets:

```bash
python -m stanza.utils.training.run_constituency it_vit --score_dev
python -m stanza.utils.training.run_constituency it_vit --score_test
```

### Silver trees

We have found that making a fake dataset of silver trees improves performance.

This is most noticeable either on small gold datasets, or on datasets
with low accuracy.  The pattern is not yet clear, since we've done it
on three datasets.  For both IT and VI, a dataset of ~8000 training
trees led to a parser with an accuracy in the low 80s, and adding a
silver dataset improved F1 by ~1.  For EN, a dataset of ~40K training
trees (PTB) led to a parser with an accuracy in the high 95.X, and
adding a silver dataset had barely any effect.  If you find a large
dataset with low accuracy, or a small dataset with high accuracy, and
help us narrow down the difference, that would be excellent!

What we do is the following.  We train 5x models which are each
slightly different, then train a second batch of 5x models with a
different transition scheme.  Each set of 5 models is used in an
ensemble to parse all of Wikipedia or some other large text repo.  We
then take all trees of length 10 to 100 where the two ensembles agree,
and use this as trees where we have high confidence in their accuracy.

To train multiple similar, but slightly different models, you can use
`--bert_hidden_layers N` to use a different number of hidden layers
from the transformer you are using (assuming you are using one), or
`--seed N` to train a model from different initial conditions

Different transition schemes can be triggered with
`--transition_scheme IN_ORDER` (default),
`--transition_scheme TOP_DOWN`, or even
`--reversed` to try parsing backwards

[Wikipedia dumps](https://dumps.wikimedia.org/backup-index-bydb.html) - look for your language code, then download "<lang>-<date>-pages-meta-current.xml.bz2"

[Wikipedia extractor](https://github.com/attardi/wikiextractor)

[Script to tokenize the extracted wikipedia](https://github.com/stanfordnlp/stanza/blob/6b9ecae54bbb5b95d42c9732675180e3aa4653d3/stanza/utils/datasets/constituency/tokenize_wiki.py) - you will need a tokenizer for your language, which may be an issue if you are working on a language Stanza does not currently know about.  Then again, you'll need a tokenizer for that language to use the parser on raw text, anyway

[Script to run an ensemble of models on tokenized text](https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/constituency/ensemble.py)

For this, you will run it as follows, except you will obviously need
to substitute different names for the models, the retagging package,
and the language, and you will want to run this on both sets of 5
models for each tokenized file produced from Wikipedia (or whatever
other data source you used):

```
python3 -m stanza.models.constituency.ensemble saved_models/constituency/en_wsj_topbert_?.pt --mode parse_text --tokenized_file AA_tokenized --retag_package en_combined_bert --lang en --predict_file AA.topdown.mrg
```

Very simple script for finding common trees in two files:

[python3 -m stanza.utils.datasets.constituency.common_trees \<file1\> \<file2\>](https://github.com/stanfordnlp/stanza/blob/6b9ecae54bbb5b95d42c9732675180e3aa4653d3/stanza/utils/datasets/constituency/common_trees.py)

So, for example, after producing tree files from the `AA` section of English Wikipedia, we did this:

```
python3 -m stanza.utils.datasets.constituency.common_trees AA.topdown.mrg AA.inorder.mrg > AA.both.mrg
```

After combining the models, we uniquify and shuffle the trees:

```
cat ??.both.mrg | sort | uniq | shuf > en.both.mrg
```

Then take the top 1M or so:

```
head -n 1000000 en.both.mrg > en_wsj.silver.mrg
```

This is now our silver training set.  As mentioned above, it was very helpful for IT and VI, and not particularly effective for EN.
You can try this after building an initial model if you want to get improved accuracy.

### Useful flags

`run_constituency` and the constituency parser main program,
`stanza.models.constituency_parser`, have several flags which may be
relevant for training and testing.

| Pretrain Option | Type | Default | Description |
| --- | --- | --- | --- |
| --wordvec_pretrain_file | str | depends on language | Instead of using the default pretrain, use this pretrain file.  Especially relevant if a language has no default pretrain |
| --no_charlm | -- | -- | Turn off the charlm entirely |
| --charlm_forward_file | str | depends on language | If you trained a charlm yourself, this will specify where the forward model is |
| --charlm_backward_file | str | depends on language | If you trained a charlm yourself, this will specify where the backward model is |
| --bert_model | str | depends on language | Which transformer to use.  Defaults are in [stanza/utils/training/common.py](https://github.com/stanfordnlp/stanza/blob/6b9ecae54bbb5b95d42c9732675180e3aa4653d3/stanza/utils/training/common.py) |
| --no_bert | -- | -- | Turn off transformers entirely |
| Training Option | Type | Default | Description |
| --epochs | int | 400 | How long to train |
| --transition_scheme | str | IN_ORDER | IN_ORDER works best, TOP_DOWN works okay, others were all experimental and didn't really help |
| --silver_file | str | -- | Which file to use for silver trees, if any |

Note that the default before for `--epochs` is to train for the first
1/2 epochs with AdaDelta, then switch to either Madgrad or AdamW for
the final 1/2 epochs.  If performance levels off for a long time, you
should not terminate training early, as there is usually quite a bit
of improvement in epochs 200-210 and possibly some smaller improvement
after 210.
