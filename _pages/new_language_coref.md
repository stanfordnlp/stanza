---
layout: page
title: Adding a new Coref model
keywords: coref, stanza, model training
permalink: '/new_language_coref.html'
nav_order: 8
parent: Training
---

**NOTE: coreference is a feature under active development; these instructions may be subject to change until release.**

## Training a new Coreference Model

Here, we present a complete end-to-end example on how to build and use your very own coreference resolution model. We will use the [CorefUD](https://ufal.mff.cuni.cz/corefud) format and data for this example, but you are free to use any other datasets which will be suitable to your use as long as they are [annotated in the CorefUD format](https://ufal.mff.cuni.cz/~popel/corefud-1.0/corefud-1.0-format.pdf).

### Grab Your Tools

- **OS**: we will use POSIX-standard tooling (Linux, macOS) for this example, but most steps can be reproduced on Windows with only difference in environment.
- **Python**: Stanza only support Python 3.8 or later.

To begin, clone our repository:

<!-- TODO CHANGE THIS TO DEV UPON RELEASE -->

```bash
git clone git@github.com:stanfordnlp/stanza.git -b dev
cd stanza
```

For the rest of the instructions here, we assume you are located within the root of the project (i.e. by following the instructions above). That is, `./` represents the root of the repo.

To get dependencies, run 

```bash
pip install .
```

if your platform has specialized hardware (in particular, GPU), take care to install the correct version of PyTorch by [following platform specific instructions](https://pytorch.org/).

### Prepare Data

Let's call our fancy new coref model `my_coref`. To get started on training, we first have to prepare and load our data. Create two folders:

```bash
mkdir -p ./extern_data/coref/
mkdir -p ./data/coref
```

Our data preparation script will prepare the data living in `./extern_data/coref` and place them into `./data/coref`.

Let's make a folder specifically for our model's data:

```bash
mkdir -p ./extern_data/coref/my_coref
```

CorefUD data lives in the form of `.connlu` files. Grab a handful of them (for instance, from the [official CorefUD 1.2 download link](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5478/CorefUD-1.2-public.zip)) you want to train on, taking care that your `train` and `dev` data are paired with the same name. Place them in `./extern_data/coref/my_coref`. That is:

```bash
$ ls -l ./extern_data/coref/my_coref
total 35328
-rw-r--r--  1 houjun  staff     73169 Mar 28 06:21 en_parcorfull-corefud-dev.conllu
-rw-r--r--  1 houjun  staff    574717 Mar 28 06:21 en_parcorfull-corefud-train.conllu
-rw-r--r--  1 houjun  staff   1940312 Mar 28 06:21 fr_democrat-corefud-dev.conllu
-rw-r--r--  1 houjun  staff  15491875 Mar 28 06:21 fr_democrat-corefud-train.conllu
```

We are ready to convert the data format! Run:

```bash
COREF_BASE=./extern_data/coref/ COREF_DATA_DIR=./data/coref python -m stanza.utils.datasets.coref.convert_udcoref --directory my_coref
```

(you can add as many `-s` flags as you'd like, for instance for a test split or ablations)

and you should see that the training data is prepared in `./data/coref`, that is:

```bash
$ ls -l ./data/coref/
total 46336
-rw-r--r--  1 houjun  staff   2488032 Jul 24 16:37 my_coref.dev.json
-rw-r--r--  1 houjun  staff  21231649 Jul 24 16:37 my_coref.train.json
```

#### New Dataset

If you are building a new dataset which is previously unsupported, you
will need to write a conversion script which turns the data into .json
files such as produced by `convert_udcoref`.  The easiest way to do this
is to build a script similar to
[stanza/utils/datasets/coref/convert_hindi.py](https://github.com/stanfordnlp/stanza/blob/af3d42b70ef2d82d96f410214f98dd17dd983f51/stanza/utils/datasets/coref/convert_hindi.py),
as this script converts a single language and should be easier to follow along.

You will need to create a [Pipeline](getting_started.md#building-a-pipeline)
for the language in question, as the model uses the heads of
dependencies to form its attachments.

Once that is created, you will want to split each document into lists
of words for each sentence, lists of coref chains for each sentence
(see the format used in the
[process_document](https://github.com/stanfordnlp/stanza/blob/af3d42b70ef2d82d96f410214f98dd17dd983f51/stanza/utils/datasets/coref/utils.py#L61)
utility function), and lists of speakers (which can be `""` if no
speakers are annotated).  The utility function `process_document` will
use the pipeline to find the heads of the coref phrases and turn the
document into a json-ready document.  You can then write this out with
the python [json](https://docs.python.org/3/library/json.html)
library, as done in `convert_hindi.py`.

One setting you may want to experiment with is the usage of `cconj`.
Being the conjunction-aware word level coref tool, this model uses
conjunctions to mark the heads of phrases.  However, while that has
worked well in the languages we've experimented with, it won't
necessarily help with all languages.

### Train Model

#### Configure Your Training

We are almost ready to train a new coref model! The next order of business involves editing the coref model configuration to your liking. To do this, first, copy the configuration file to a good target destination:

```bash
cp ./stanza/models/coref/coref_config.toml ./data/coref_config.toml
```

also, make a folder for where our trained model weights should go:

```bash
mkdir -p saved_models/coref
```

Now, let us edit our config, open up your `./data/coref_config.toml` and set the following variables:

```toml
train_data = "data/coref/my_coref.train.json"
dev_data = "data/coref/my_coref.dev.json"
test_data = "data/coref/my_coref.dev.json"
```

Read through the .toml for any changes to the architecture you would like to make. Our recommendations are given as the default. In particular, pay attention to line 137 and later, which contains information about each of the backbone architectures you can choose to use, and whether you want to tune them with full fine tuning or Low-Rank Approximations (LoRA).

Each of these configurations are called an **experiment**. For instance, the *xlm_roberta_lora* experiment (which is the standard multilingual setup of our released model) contains the following configuration:

```toml
[xlm_roberta_lora]
bert_model = "FacebookAI/xlm-roberta-large"
bert_learning_rate = 0.000025
lora = true
lora_target_modules = [ "query", "value", "output.dense", "intermediate.dense" ]
lora_modules_to_save = [ "pooler" ]
```

This tells our system that, in this experiment, we want to use Facebook's Roberta-Lora as backbone, with a learning rate of `2.5e-5`, and we want to use low-rank approximations.

#### Training!
Once you feel ready to go, it's time to train the model. To do this, run:

```bash
python -m stanza.models.wl_coref train [your experiment] --config_file ./data/coref_config.toml
```

For instance, to run the *xlm_roberta_lora* experiment above, run:

```bash
python -m stanza.models.wl_coref train xlm_roberta_lora --config_file ./data/coref_config.toml
```

If you would like to use experiment tracking with Weights and Biases, run:

```bash
python -m stanza.models.wl_coref train [your experiment] --config_file ./data/coref_config.toml --wandb
```

#### Scoring

Once your model has converged, run:

```bash
python -m stanza.models.wl_coref eval [your experiment] --config_file ./data/coref_config.toml
```

to score on the configured dev set, and 

```bash
python -m stanza.models.wl_coref eval [your experiment] --data-split test --config_file ./data/coref_config.toml
```

to score on the test set.

### Using Local Models

Of course, once you trained your model, you want to use it as a part of the coreference pipeline! This is fairly simple. Locate your shiny new weights at `./saved_models/coref` (should be `xlm_roberta_lora.pt` if you followed our example); then, run:

```python
>>> import stanza
>>> nlp = stanza.Pipeline('en', processors='tokenize,mwt,coref', coref_model_path="./saved_models/coref/xlm_roberta_lora.pt")
>>> nlp("I study coreferences, and I love them too!")
```

Congratulations! You have trained and deployed your own coref model.
