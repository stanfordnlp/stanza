---
title: Pipeline and Processors
keywords: pipeline
permalink: '/pipeline.html'
---

## Pipeline

Users of Stanza can process documents by building a [`Pipeline`](pipeline.md#pipeline) with the desired [`Processor`](pipeline.md#processors) units.  The pipeline takes in a [`Document`](data_objects.md#document)
object or raw text, runs the processors in succession, and returns an annotated [`Document`](data_objects.md#document).

You can customize the pipeline by specifying the options in the table below:

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| lang | str | "en" | Language code for the language to process with the Pipeline.  See [here](models.md) for a complete list of available languages. |
| dir | str | ~/stanza_resources | Directory for storing the models. |
| package | str | "default" | Package to use for processors. See [here](models.md) for a complete list of available packages. |
| processors | dict or str | {} | [Processor](pipeline.md#processors)s to use in the Pipeline. If str, should be comma-seperated processor names to use (e.g., 'tokenize,pos'). If dict, should specify the processor name with its package (e.g., {'tokenize': package, 'pos': package}).  |
| logging_level | str | 'INFO' | Control the details of information to display. Can be one of 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CIRTICAL', 'FATAL'. Less information will be displayed from 'DEBUG' to 'FATAL'. |
| verbose | str | None | Simplified option for logging level. If True, set logging level to 'INFO'. If False, set logging level to 'ERROR'.  |
| use_gpu | bool | True | Attempt to use a GPU if possible. |
| kwargs | - | - | Options for each of the individual processors. See the individual processor pages for descriptions. |

## Processors

Processors are units of the neural pipeline that create different annotations for a [`Document`](data_objects.md#document). The neural pipeline now supports the following processors:

| Name | Annotator class name | Generated Annotation | Description |
| --- | --- | --- | --- | 
| tokenize | TokenizeProcessor | Segments a [`Document`](data_objects.md#document) into [`Sentence`](data_objects.md#sentence)s, each containing a list of [`Token`](data_objects.md#token)s. This processor also predicts which tokens are multi-word tokens, but leaves expanding them to the [MWT expander](mwt.md). | Tokenizes the text and performs sentence segmentation. |
| mwt | MWTProcessor | Expands multi-word tokens into multiple words when they are predicted by the tokenizer. | Expands [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html) predicted by the [tokenizer](tokenize.md). |
| lemma | LemmaProcessor | Perform [lemmatization](https://en.wikipedia.org/wiki/Lemmatisation) on a [`Word`](data_objects.md#word) using the `Word.text` and `Word.upos` value. The result can be accessed in `Word.lemma`. | Generates the word lemmas for all tokens in the corpus. |
| pos | POSProcessor | UPOS, XPOS, and UFeats annotations accessible through [`Word`](data_objects.md#word)'s properties `pos`, `xpos`, and `ufeats`. | Labels tokens with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html). |
| depparse | DepparseProcessor | Determines the syntactic head of each word in a sentence and the dependency relation between the two words that are accessible through [`Word`](data_objects.md#word)'s `head` and `deprel` attributes. | Provides an accurate syntactic dependency parser. |
| ner | NERProcessor | Named entities accessible through [`Document`](data_objects.md#document) or [`Sentence`](data_objects.md#sentence)'s properties `entities` or `ents`. Token-level NER tags accessible through [`Token`](data_objects.md#token)'s properties `ner`. | Recognize named entities for all token spans in the corpus. |

## Usage

### Basic Example

You can easily build the pipeline with options specified above:

```python
import stanza

nlp = stanza.Pipeline("en", processors='tokenize,pos', use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
doc = nlp("Barack Obama was born in Hawaii.") # Run the pipeline on the input text
print(doc) # Look at the result
```

You can find more intutive examples about how to use these options [here](installation_usage.md#model-downloading).

### Build Pipeline from Config 

Alternatively, you can build the desired pipeline with a config, allowing maximum customization for the pipeline:

```python
import stanza

config = {
	'processors': 'tokenize,mwt,pos', # Comma-separated list of processors to use
	'lang': 'fr', # Language code for the language to build the Pipeline in
	'tokenize_model_path': './fr_gsd_models/fr_gsd_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
	'mwt_model_path': './fr_gsd_models/fr_gsd_mwt_expander.pt',
	'pos_model_path': './fr_gsd_models/fr_gsd_tagger.pt',
	'pos_pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt',
	'tokenize_pretokenized': True # Use pretokenized text as input and disable tokenization
}
nlp = stanza.Pipeline(**config) # Initialize the pipeline using a configuration dict
doc = nlp("Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie .") # Run the pipeline on the pretokenized input text
print(doc) # Look at the result
```

