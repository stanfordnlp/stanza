---
title: Pipeline
keywords: pipeline
permalink: '/pipeline.html'
---

## Pipeline

Users of StanfordNLP can process documents by building a `Pipeline` with the desired `Processor` units.  The pipeline takes in a `Document`
object or raw text, runs the processors in succession, and returns a fully annotated `Document`.

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| lang | str | 'en' | Use recommended models for this language |
| models_dir | str | ~/stanfordnlp_resources | Directory for storing the models |
| processors | str | 'tokenize,mwt,pos,lemma,depparse' | List of processors to use |
| treebank | str | None | Use models for this treebank |
| use_gpu | bool | True | Attempt to use a gpu if possible |

Options for each of the individual processors can be specified when building the pipeline.  See the individual
processor pages for descriptions.

## Usage

### Basic Example

```python
import stanfordnlp

stanfordnlp.download('en') # download the English models
nlp = stanfordnlp.Pipeline(processors='tokenize,ssplit,pos', models_dir='/path/to/stanfordnlp_resources', treebank='en_ewt', use_gpu=True, pos.batch_size=3000) # build the pipeline, specify part-of-speech processor's batch size
doc = nlp("Barack Obama was born in Hawaii.") # run the pipeline on input text
```

### Specifying A Full Config 

```python
import stanfordnlp

config = {'processors': 'tokenize,mwt,pos,lemma,depparse', 'lang': 'fr', 'tokenize.model_path': './fr_gsd_models/fr_gsd_tokenizer.pt', 'mwt.model_path': './fr_gsd_models/fr_gsd_mwt_expander.pt', 'pos.model_path': './fr_gsd_models/fr_gsd_tagger.pt', 'pos.pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt', 'lemma.model_path': './fr_gsd_models/fr_gsd_lemmatizer.pt', 'depparse.model_path': './fr_gsd_models/fr_gsd_parser.pt', 'depparse.pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt'}
nlp = stanfordnlp.Pipeline(**config)
doc = nlp("Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie.")
```

