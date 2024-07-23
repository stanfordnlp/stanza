---
layout: page
title: Retrain models for an NER dataset
keywords: stanza, ner model training
permalink: '/retrain_ner.html'
nav_order: 11
parent: Training
---

## End to End Retraining Example

Retraining a single NER model is going to be much simpler than
building the model from scratch, as the code to convert the data is
already available in Stanza.

Here we explain how to replicate the experiments in the forthcoming
Worldwide NER dataset paper, "Do “English” Named Entity Recognizers Work Well on Global Englishes?"

### Environment

First, please
[set up your environment](https://stanfordnlp.github.io/stanza/new_language_ner.html#environment)
in the same manner as for building a completely new NER model.

### Download data

For most of the datasets, you will need to manually download the data.
Instructions for each dataset Stanza knows how to process are at the
top of the `prepare_ner_dataset.py` script:

https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/datasets/ner/prepare_ner_dataset.py

For example, there are specific (and very simple) instructions on where to download the Worldwide dataset:

https://github.com/stanfordnlp/stanza/blob/5d5a4c35f2147601869a68e7b4c22a3274488996/stanza/utils/datasets/ner/prepare_ner_dataset.py#L348

Going forward, a few datasets will automatically download as
necessary.  For example, starting with Stanza 1.7.0 (currently in the
`dev` branch of the Stanza repo), English CoNLL03 will be
automatically downloaded as part of the data preparation script.

### Prepare data

To prepare the data, it will need to be converted to a `.json` format used by Stanza.  The
[`prepare_ner_dataset`](https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/datasets/ner/prepare_ner_dataset.py)
script is capable of doing that for most of the NER models released in Stanza.

For English CoNLL03, simply do:

```
python3 stanza/utils/datasets/ner/prepare_ner_dataset.py en_conll03
```

The Worldwide paper frequently references a conversion of Worldwide to
the 4 classes used in CoNLL.  This is also part of the prepare script:

```
python3 stanza/utils/datasets/ner/prepare_ner_dataset.py en_worldwide-4class
```

Finally, there is a combined method which first prepares both of the
datasets, then combines the training data into one file:

```
python3 stanza/utils/datasets/ner/prepare_ner_dataset.py en_conll03ww
```

### Train model

Once the data is prepared, the models can be trained using a command line such as:

```
python3 stanza/utils/training/run_ner.py en_conll03
python3 stanza/utils/training/run_ner.py en_worldwide-4class
python3 stanza/utils/training/run_ner.py en_conll03ww
```

By default, this will not use any transformer.  A default transformer can be used with the `--use_bert` flag, such as

```
python3 stanza/utils/training/run_ner.py en_conll03 --use_bert
```

To use a specific transformer (must be available on HF), use the `--bert_model` flag:

```
python3 stanza/utils/training/run_ner.py en_conll03 --use_bert --bert_model roberta-large
```

If a model already exists, `run_ner.py` will not clobber that model.  You can force it to clobber with the `--force` flag:

```
python3 stanza/utils/training/run_ner.py en_conll03 --use_bert --bert_model roberta-large --force
```

### Test model

To test the model, you can use the `--score_dev` or `--score_test` flags as appropriate.

One thing to note is that the `run_ner.py` script will build the model
filename taking into account the embedding used.  There are two
choices for making sure you are testing the right model.  You can
either provide the exact filename of the model to use for testing, or
you can provide the same transformer flags as used when training:

```
python3 stanza/utils/training/run_ner.py en_conll03 --score_dev --use_bert --bert_model roberta-large
python3 stanza/utils/training/run_ner.py en_conll03 --score_dev --save_name saved_models/ner/en_conll03_roberta-large_nertagger.pt
```

If you don't like the default save names, you can also use the `--save_name` flag when training a model.

### Testing across datasets

You can test models on a different dataset with the `--eval_file` flag.  For example, to use a Worldwide test file on the CoNLL03 model:

```
python3 stanza/utils/training/run_ner.py en_conll03 --score_dev --use_bert --bert_model roberta-large --eval_file data/ner/en_worldwide-4class.test.json
```

There is one weird caveat here that it is necessary to use the same
dataset name as used when training, as sometimes the word vectors or
character model used will be specifically chosen for the model in
question.

When running experiments for the paper, we standardized which word
vectors we used across test cases.  Indeed, it is possible to
replicate that feature as well.  Simply supply the `--wordvec_pretrain_file` flag when both training and testing:

```
python3 stanza/utils/training/run_ner.py en_conll03 --use_bert --bert_model roberta-large --wordvec_pretrain_file ~/stanza_resources/en/pretrain/conll17.pt
python3 stanza/utils/training/run_ner.py en_conll03 --score_dev --use_bert --bert_model roberta-large --wordvec_pretrain_file ~/stanza_resources/en/pretrain/conll17.pt
```

