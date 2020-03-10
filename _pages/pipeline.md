---
title: Pipeline and Processors
keywords: pipeline
permalink: '/pipeline.html'
---

## Pipeline

To start annotating text with Stanza, you would typically start by building a [`Pipeline`](pipeline.md#pipeline) that contains [`Processor`](pipeline.md#processors)s, each fulfilling a specific NLP task you desire (e.g., tokenization, part-of-speech tagging, syntactic parsing, etc). The pipeline takes in raw text or a [`Document`](data_objects.md#document) object that contains partial annotations, runs the specified processors in succession, and returns an annotated [`Document`](data_objects.md#document) (see the documentation on [`Document`](data_objects.md#document) for more information on how to extract these annotations).

To build and customize the pipeline, you can specify the options in the table below:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| lang | `str` | `'en'` | Language code (e.g., `"en"`) or language name (e.g., `"English"`) for the language to process with the Pipeline. You can find a complete list of available languages [here](models.md).  |
| dir | `str` | `"~/stanza_resources"` | Directory for storing the models downloaded for Stanza. By default, Stanza stores its models in a folder in your home directory. |
| package | `str` | `'default'` | Package to use for processors, where each package typically specifies what data the models are trained on. We provide a "default" package for all languages that contains NLP models most users will find useful. A complete list of available packages can be found [here](models.md). |
| processors | `dict` or `str` | `dict()` | [Processor](pipeline.md#processors)s to use in the Pipeline. This can either be specified as a comma-seperated list of processor names to use (e.g., `'tokenize,pos'`), or a Python dictionary with Processor names as keys and packages as corresponding values (e.g., `{'tokenize': 'ewt', 'pos': 'ewt'}`). All unspecified Processors will fall back to using the package specified by the `package` argument. A list of all Processors supported can be found [here](pipeline.md#processors) |
| logging_level | `str` | `'INFO'` | Controls the level of logging information to display when the Pipeline is instantiated and run. Can be one of `'DEBUG'`, `'INFO'`, `'WARN'`, `'ERROR'`, `'CIRTICAL'`, or `'FATAL'`. Less and less information will be displayed from `'DEBUG'` to `'FATAL'`. |
| verbose | `str` | `None` | Simplified option for logging level. If `True`, logging level will be set to `'INFO'`. If `False`, logging level will be set to `'ERROR'`.  |
| use_gpu | `bool` | `True` | Attempt to use a GPU if available. Set this to `False` if you are in a GPU-enabled environment but want to explicitly keep Stanza from using the GPU. |
| kwargs | - | - | Options for each of the individual processors. See the individual processor pages for descriptions. |

## Processors

Processors are units of the neural pipeline that perform specific NLP functions and create different annotations for a [`Document`](data_objects.md#document). The neural pipeline supports the following processors:

| Name | Processor class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| tokenize | TokenizeProcessor | - | Segments a [`Document`](data_objects.md#document) into [`Sentence`](data_objects.md#sentence)s, each containing a list of [`Token`](data_objects.md#token)s. This processor also predicts which tokens are multi-word tokens, but leaves expanding them to the [MWTProcessor](mwt.md). | Tokenizes the text and performs sentence segmentation. |
| mwt | MWTProcessor | tokenize | Expands multi-word tokens (MWTs) into multiple words when they are predicted by the tokenizer. Each [`Token`](data_objects.md#token) will correspond to one or more [`Word`](data_objects.md#word)s after tokenization and MWT expansion. | Expands [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html) predicted by the [TokenizeProcessor](tokenize.md). This is only applicable to some languages. |
| pos | POSProcessor | tokenize, mwt | UPOS, XPOS, and UFeats annotations are accessible through [`Word`](data_objects.md#word)'s properties `pos`, `xpos`, and `ufeats`. | Labels tokens with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html). |
| lemma | LemmaProcessor | tokenize, mwt, pos | Perform [lemmatization](https://en.wikipedia.org/wiki/Lemmatisation) on a [`Word`](data_objects.md#word) using the `Word.text` and `Word.upos` values. The result can be accessed as `Word.lemma`. | Generates the word lemmas for all words in the Document. |
| depparse | DepparseProcessor | tokenize, mwt, pos, lemma | Determines the syntactic head of each word in a sentence and the dependency relation between the two words that are accessible through [`Word`](data_objects.md#word)'s `head` and `deprel` attributes. | Provides an accurate syntactic dependency parsing analysis. |
| ner | NERProcessor | tokenize, mwt | Named entities accessible through [`Document`](data_objects.md#document) or [`Sentence`](data_objects.md#sentence)'s properties `entities` or `ents`. Token-level NER tags accessible through [`Token`](data_objects.md#token)'s properties `ner`. | Recognize named entities for all token spans in the corpus. |

## Usage

Using Stanza's neural Pipeline to annotate your text can be as simple as a few lines of Python code. Here we provide two simple examples, and refer the user to [our tutorials](tutorials.md) for further details on how to use each Processor.

### Basic Example

To annotate a piece of text, you can easily build the Stanza Pipeline with options introduced above:

```python
import stanza

nlp = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
doc = nlp("Barack Obama was born in Hawaii.") # Run the pipeline on the input text
print(doc) # Look at the result
```

Here, we are building a Pipeline for English that performs tokenization, sentence segmentation, and POS tagging that runs on the GPU, and POS tagging is limited to processing 3000 words at one time to avoid excessive GPU memory consumption.

You can find more examples about how to use these options [here](installation_usage.md#building-a-pipeline).

### Build Pipeline from a Config Dictionary

When there are many options you want to configure, or even set programmatically, it might not be convenient to set them one by one using keyword arguments to instantiate the Pipeline. In these cases, alternatively, you can build the desired pipeline with a config dictionary, allowing maximum customization for the pipeline:

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

Here, we can specify the language, processors, and paths for many Processor models all at once, and pass that to the Pipeline initializer. Note that config dictionaries and keyword arguments can be combined as well, to maximize your flexibility in using Stanza's neural pipeline.
