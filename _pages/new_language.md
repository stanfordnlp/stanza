---
layout: page
title: Add a New Language
keywords: stanza, model training
permalink: '/new_language.html'
nav_order: 6
parent: Usage
---

## Summary

In order to train a stanza package for a new language, you will need
data for the various models, word vectors for all models other than
tokenization, and possibly a character model for improving
performance.

## Data format

Most of the training scripts expect data in [conllu
format](https://universaldependencies.org/format.html).
Each word has its own individual line.  For examples of the data
format expected, you can download a package such as en-ewt from
https://universaldependencies.org/#download and run it through the
preprocessing scripts such as `prep_tokenize.sh`.  If your data is not
already compatible with this format, you would need to write your own
processing script to convert it to this format.

Note that many of the columns in conllu format may not be present in
your data.  Most of these columns can be represented with a blank "_".
One exception to this is the dependency column, which occupies the
7th, 8th, and 9th, columsn of the data.  There is some numeric
processing involved in these columns, so "_" is not sufficient.  If
these columns are not prsent, you should fake them as follows: set the
first row's values to `0, root, 0:root`, set each other row `i` to
`i-1, dep, i-1:dep`.  You can look at process_orchid.py for an
example.

The classifier model (which is used for sentiment) has a different
data format.  For this model, the format is one line per sentence or
phrase, with the label first and the text as a whitespaced tokenized
sentencse after that.  For example, see any of the sentiment
processing scripts.


## Word Vectors

In general we use either word2vec or fasttext word vectors.  If none
of those are available for the language you want to work on, you might
try to use [GloVe](https://github.com/stanfordnlp/GloVe) to train your
own word vectors.

More information on using word vectors [here](word_vectors.md).

## Character LM

Character LMs are included for a few languages.  You can look in
resources.json for `forward_charlm` and `backward_charlm`

For adding a new languages, we provide scripts to automate large parts of the process.  Scripts for converting raw text to conllu and conllu to a charlm dataset can be found in [https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/charlm/conll17_to_text.py](conll17_to_text.py) and [https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/charlm/make_lm_data.py](make_lm_data.py)

* Gather a ton of tokenized text.  Ideally gigabytes.  Wikipedia is a good place to start for raw text, but in that case you will need to tokenize it.
* If the data you gathered was from the conll17 shared task, we provide a script to turn it into txt files.  Run ```python3 stanza/utils/charlm/conll17_to_text.py ~/extern_data/finnish/conll17/Finnish/```  This will convert conllu or conllu.xz files to txt and put them in the same directory.
* Run ```python3 stanza/utils/charlm/make_lm_data.py extern_data/charlm_raw extern_data/charlm```  This will convert text files in the `charlm_raw` directory to a suitable dataset in `extern_data/charlm`.  You may need to adjust your paths.
* Run ```python3 -m stanza.models.charlm --train_dir extern_data/charlm/fi/conll17/train --eval_file extern_data/charlm/fi/conll17/dev.txt --direction forward --lang fi --shorthand fi_conll17  --mode train```
* This will take days or weeks to fully train.

## Building models

Once you have the needed resources, you can follow the instructions
[here](https://stanfordnlp.github.io/stanza/training.html) to train
the models themselves.

## Integrating into Stanza

Once you have trained new models, you need to integrate your models
into the available resources.

The stanza models are kept in your `stanza_resources` directory, which
by default is kept in `~/stanza_resources`.  A json description of the
models is needed so that stanza knows which models are prerequisites
for other models.

The problem with editing this directly is that if you download more
officially released models from stanza, any changes you make will be
overwritten.  A solution to this problem is to make your own directory
with a new json file.  For example, if you were to create new Thai
tokenizers, you could make a directory `thai_stanza_resources` with a
file `resources.json` in it.  You could copy a block with information
for the models:

```
{
  "th": {
    "tokenize": {
      "orchid": {
      },
      "best": {
      }
    },
    "default_processors": {
      "tokenize": "orchid"
    },
    "default_dependencies": {
    },
    "lang_name": "Thai"
  }
}
```

The resources directory then needs a structure where the first
subdirectory is the language code, so in this case
`/home/username/thai_resources/th`.  Each model type then gets a
further subdirectory under that directory.  For example,
the `orchid` tokenizer model goes in
`/home/username/thai_resources/th/tokenize/orchid.pt`
and the `best` tokenizer model goes in
`/home/username/thai_resources/th/tokenize/best.pt`

At last, you can load the models via

```
import stanza
pipeline = stanza.Pipeline("th", dir="/home/username/thai_resources")
```

[There are several options for configuring a new pipeline and its use of resources](https://stanfordnlp.github.io/stanza/pipeline.html)
You can see the existing `resources.json` for examples of how to build
the json entries for other models.

## Contributing Back to Stanza

If you feel your finished model would be useful for the wider
community, please feel free to share it back with us!  We will
evaluate it and include it in our distributions if appropriate.

Please describe the data sources used and any options used or
modifications made so that the models can be recreated as needed.

You can open [an issue](https://github.com/stanfordnlp/stanza/issues)
on our main page.  For example, the Ukrainian NER model we have
[was provided via an issue](https://github.com/stanfordnlp/stanza/issues/319).

