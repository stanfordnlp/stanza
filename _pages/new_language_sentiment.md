---
layout: page
title: Adding a new Sentiment model
keywords: sentiment, stanza, model training
permalink: '/new_language_sentiment.html'
nav_order: 9
parent: Usage
---

## End to End Sentiment example

Starting with the next release of Stanza (1.4.1), there will be a new mechanism for training Sentiment models.

Here is a complete end to end example on how to build a Sentiment model for a previously unknown language.  For this example, we will use a Spanish dataset:

http://tass.sepln.org/2020/?page_id=74

There are multiple tasks on that page.  Task 2 does not have gold annotations (as of July 2022), making it an easy choice of Task 1.


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
git checkout -b spanish_sentiment
```

### Environment

There are many environment variables mentioned in the usage page,
along with a `config.sh` script which can set them up.  However,
ultimately only two are relevant for a Sentiment model,
`$SENTIMENT_BASE` and `$SENTIMENT_DATA_DIR`.

Both of these have reasonable defaults, but we can still customize them.

`$SENTIMENT_BASE` determines where the *raw, unchanged* datasets go.

The purpose of the data preparation scripts will be to put *processed*
forms of this data in `$SENTIMENT_DATA_DIR`.  Once this is done, the
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
export SENTIMENT_BASE=/u/nlp/data/sentiment/stanza
export SENTIMENT_DATA_DIR=/nlp/scr/$USER/data/ner
```

Since you will be running python programs directly from the git checkout of Stanza, you will need to make sure `.` is in your `PYTHONPATH`.

### Language code

If your dataset involves a language currently not in Stanza, it may
need [the language code](new_language.md#language-codes) added to
Stanza.  Feel free to open an issue on our github or send us a PR.

### Data download

There are several files on the Tass 2020 site which refer to Task 1 (1.1, 1.2).  We download them all to a directory:

```
$SENTIMENT_BASE/spanish/tass2020
```

It will not be necessary to unzip them, as the script will process the zip files directly.

For a new sentiment dataset, please arrange to have it downloaded to `$SENTIMENT_BASE/<lang>/...`

### Processing raw data to .json

The Sentiment model uses a .json format for storing text and
annotations.  It may not be strictly necessary to use .json, but this
gives us an easy way to store tokens with spaces (such as can be found
in Vietnamese).

Unfortunately, unlike NER datasets or especially the UD ecosystem of
datasets, there is no standard format for sentiment datasets.  What
you will need to do is write code to turn the dataset into 3 lists of
labels and text.  An example can be seen in this PR:

https://github.com/stanfordnlp/stanza/pull/1104

The code to translate the test set is here:

[Code to read and process the TASS2020 test set](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/process_es_tass2020.py#L147)

The train & dev sets are here:

[Code to read and process the TASS2020 train and dev sets](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/process_es_tass2020.py#L158)

These code paths both read the `.zip` files in the dataset and turn
them into lists of labels and text.  Originally the text is raw
strings, untokenized, but we then turn it into lists of words.

Internally, the sentiment tool uses word vectors as a base input
layer.  It is also possible to use the pretrained charlm or a
transformer.  Because of the use of word vectors, though, it is
expected that the text be tokenized into listss of words.

[Code to tokenize the TASS2020 text](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/process_es_tass2020.py#L131)

All languages for which we have any support already have a tokenizer.
An unknown language with no tokenizer will need a separate mechanism
to handle this.

{% include alerts.html %}
{{ note }}
{{ "[Only load the Pipeline once](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/process_es_tass2020.py#L193), then pass it around.  Otherwise, loading will be very expensive." | markdownify }}
{{ end }}

To write the code to the `.json` format used by the training tool, there is a
[`SentimentDatum` tuple](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/process_utils.py#L13)
which you can use to store a single item,
a [`write_list`](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/process_utils.py#L18) function for writing a list of `SentimentDatum`,
and a [`write_dataset`](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/process_utils.py#L41) function which writes three lists, train, dev, and test.
Please refer to `process_es_tass2020.py` for examples of how to use them.

Once the code to translate the dataset is written, we add some
documentation and a function call to
`stanza/utils/datasets/ner/prepare_sentiment_dataset.py` to keep
everything organized in one place.

[Documentation example](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/prepare_sentiment_dataset.py#L160)

[Function to call the conversion](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/prepare_sentiment_dataset.py#L341)

[Entry connecting the function](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/prepare_sentiment_dataset.py#L360) in the [dataset mapping](https://github.com/stanfordnlp/stanza/blob/906ea5f6188942fdd8a5a276c5457a8161a3e7ee/stanza/utils/datasets/sentiment/prepare_sentiment_dataset.py#L349)

The intention is that once the script is prepared, the new dataset has been added to the general preparation script:

```
python3 stanza.utils.datasets.sentiment.prepare_sentiment_dataset es_tass2020
```

### Labels

Typically we use `0`, `1`, and `2` to represent `Negative`, `Neutral`,
and `Positive` in three class sentiment tasks.  This is not necessary,
though; the tool can process any number of labels, and it can also
process labels which are not numeric!

{% include alerts.html %}
{{ note }}
{{ "Technically the tool can be used for any sentence classification task, not just sentiment." | markdownify }}
{{ end }}

### Word Vectors

The base version of the tool uses word vectors as an input layer to the classifier.

Please refer to the word vectors section of [NER models](new_language_ner.md#word-vectors) to add a new set of word vectors.

{% include alerts.html %}
{{ note }}
{{ "Stanza already has Spanish word vectors, so we do not need to do anything to add them.  In fact, the `run_sentiment` script will attempt to automatically download them." | markdownify }}
{{ end }}

### Training!

At this point, everything is ready to push the button and start training.

```bash
python -m stanza.utils.training.run_sentiment es_tass2020
```


### Contributing back

If you like, you can open a PR with your code changes and post the
models somewhere we can integrate them.  It would be very appreciated!

### Citations

TODO

