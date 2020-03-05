---
title: Pipeline
keywords: pipeline
permalink: '/pipeline.html'
---

## Pipeline

Users of StanfordNLP can process documents by building a [`Pipeline`](pipeline.md) with the desired `Processor` units.  The pipeline takes in a [`Document`](data_objects.md#document)
object or raw text, runs the processors in succession, and returns an annotated [`Document`](data_objects.md#document).

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| lang | str | "en" | Language code for the language to process with the Pipeline.  See [here](models.md) for a complete list of available languages. |
| dir | str | ~/stanfordnlp_resources | Directory for storing the models. |
| package | str | "default" | Package to use for processors. See [here](models.md) for a complete list of available packages. |
| processors | dict or str | {} | [Processor](processors.md)s to use in the Pipeline. If str, should be comma-seperated processor names to use (e.g., 'tokenize,pos'). If dict, should specify the processor name with its package (e.g., {'tokenize': package, 'pos': package}).  |
| logging_level | str | 'INFO' | Control the details of information to display. Can be one of 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CIRTICAL', 'FATAL'. Less information will be displayed from 'DEBUG' to 'FATAL'. |
| verbose | str | None | Simplified option for logging level. If True, set logging level to 'INFO'. If False, set logging level to 'ERROR'.  |
| use_gpu | bool | True | Attempt to use a GPU if possible. |
| kwargs | - | - | Options for each of the individual processors. See the individual processor pages for descriptions. |

## Usage

### Basic Example

You can easily build the pipeline with options specified above:

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline("en", processors='tokenize,pos', use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
doc = nlp("Barack Obama was born in Hawaii.") # Run the pipeline on the input text
print(doc) # Look at the result
```

You can find more intutive examples about how to use these options [here](installation_usage.md#model-downloading).

### Build Pipeline from Config 

Alternatively, you can build the desired pipeline with a config, allowing maximum customization for the pipeline:

```python
import stanfordnlp

config = {
	'processors': 'tokenize,mwt,pos', # Comma-separated list of processors to use
	'lang': 'fr', # Language code for the language to build the Pipeline in
	'tokenize_model_path': './fr_gsd_models/fr_gsd_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
	'mwt_model_path': './fr_gsd_models/fr_gsd_mwt_expander.pt',
	'pos_model_path': './fr_gsd_models/fr_gsd_tagger.pt',
	'pos_pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt',
	'tokenize_pretokenized': True # Use pretokenized text as input and disable tokenization
}
nlp = stanfordnlp.Pipeline(**config) # Initialize the pipeline using a configuration dict
doc = nlp("Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie .") # Run the pipeline on the pretokenized input text
print(doc) # Look at the result
```

