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
| lang | str | "en" | Use recommended models for this language. |
| models_dir | str | ~/stanfordnlp_resources | Directory for storing the models. |
| processors | str | "tokenize,mwt,pos,lemma,depparse" | List of processors to use. For a list of all processors supported, see [Processors Summary](/processors.html). |
| treebank | str | None | Use models for this treebank. If not specified, `Pipeline` will look up the default treebank for the language requested. |
| use_gpu | bool | True | Attempt to use a GPU if possible. |

Options for each of the individual processors can be specified when building the pipeline.  See the individual processor pages for descriptions.

## Usage

### Basic Example

```python
import stanfordnlp

MODELS_DIR = '.'
stanfordnlp.download('en', MODELS_DIR) # Download the English models
processor_configs = { "pos.batch_size": 3000 } # Specify part-of-speech processor's batch size. Note we can't pass this in as a "normal" Python keyword argument because of the presence of the dot
nlp = stanfordnlp.Pipeline(processors='tokenize,pos', models_dir=MODELS_DIR, treebank='en_ewt', use_gpu=True, **processor_configs) # Build the pipeline
doc = nlp("Barack Obama was born in Hawaii.") # Run the pipeline on input text
```

### Specifying A Full Config 

```python
import stanfordnlp

config = {
	'processors': 'tokenize,mwt,pos,lemma,depparse', # Comma-separated list of processors to use
	'lang': 'fr', # Language code for the language to build the Pipeline in
	'tokenize.model_path': './fr_gsd_models/fr_gsd_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}.{argument_name}"
	'mwt.model_path': './fr_gsd_models/fr_gsd_mwt_expander.pt',
	'pos.model_path': './fr_gsd_models/fr_gsd_tagger.pt',
	'pos.pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt',
	'lemma.model_path': './fr_gsd_models/fr_gsd_lemmatizer.pt',
	'depparse.model_path': './fr_gsd_models/fr_gsd_parser.pt',
	'depparse.pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt'
}
nlp = stanfordnlp.Pipeline(**config)
doc = nlp("Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie.")
```

