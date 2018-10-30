# Universal Dependencies from Scratch for Many Languages

The Stanford submission for the CoNLL-2018 Shared Task.

### Setup

The system requires Python 3.6 or greater.  

#### Requirements 

```
torch 0.4.1
```

#### Instructions

1. In the root directory, set up a directory called `extern_data`.  This must include the CoNLL18 treebank data and word2vec embeddings.

* You can access the CoNLL18 data here: http://universaldependencies.org
* You can access the word2vec embeddings here: ??


The provided scripts will expect this directory structure:

```
extern_data
|-- CoNLL18
|-- word2vec
```

2. Create a `data` directory and subdirectories for modules (e.g. depparse) you want to create.

```
mkdir -p data/depparse
```

3. Set up `scripts/config.sh` appropriately.

### Training

All training commands follow the basic format of `bash scripts/run_{}.sh TREEBANK GPU_NUM`

For example, to train the English-EWT dependency parser, run this command from the root directory:

```
bash scripts/run_depparse.sh UD_English-EWT 0
```

This will train the model with standard settings.  When training is completed a score on the dev set will be reported as well.

The necessary data files will be stored in `data/depparse` and the model will be stored in `saved_models/depparse`

### Evaluation

You can evaluate models on all the tasks with `scripts/eval.sh`

```
bash scripts/eval.sh TREEBANK TASK DATASET GPU_NUM
``` 

For example, to evaluate the part-of-speech tagger on the UD_English-EWT test set, with gpu 0, run:

```
bash scripts/eval.sh UD_English-EWT pos test 0
```
