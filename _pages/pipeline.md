---
layout: page
title: Pipeline and Processors
keywords: pipeline
permalink: '/pipeline.html'
nav_order: 1
parent: Neural Pipeline
---

## Pipeline

To start annotating text with Stanza, you would typically start by building a [`Pipeline`](pipeline.md#pipeline) that contains [`Processor`](pipeline.md#processors)s, each fulfilling a specific NLP task you desire (e.g., tokenization, part-of-speech tagging, syntactic parsing, etc). The pipeline takes in raw text or a [`Document`](data_objects.md#document) object that contains partial annotations, runs the specified processors in succession, and returns an annotated [`Document`](data_objects.md#document) (see the documentation on [`Document`](data_objects.md#document) for more information on how to extract these annotations).

To build and customize the pipeline, you can specify the options in the table below:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| lang | `str` | `'en'` | Language code (e.g., `"en"`) or language name (e.g., `"English"`) for the language to process with the Pipeline. You can find a complete list of available languages [here](models.md).  |
| dir | `str` | `'~/stanza_resources'` | Directory for storing the models downloaded for Stanza. By default, Stanza stores its models in a folder in your home directory. |
| package | `dict`&nbsp;or&nbsp;`str` | `'default'` | Package to use for processors, where each package typically specifies what data the models are trained on. We provide a "default" package for all languages that contains NLP models most users will find useful. A complete list of available packages can be found [here](models.md).  If a `dict` is used, any processor not specified will be `default`.  [ner allows for a list of packages](ner.md#using-multiple-models) in the `dict`. |
| processors | `dict`&nbsp;or&nbsp;`str` | `dict()` | [Processor](pipeline.md#processors)s to use in the Pipeline. This can either be specified as a comma-seperated list of processor names to use (e.g., `'tokenize,pos'`), or a Python dictionary with Processor names as keys and packages as corresponding values (e.g., `{'tokenize': 'ewt', 'pos': 'ewt'}`). In the case of a dict, all unspecified Processors will fall back to using the package specified by the `package` argument. To ensure that only the processors you want are loaded when using a dict, set `package=None` as well.  A list of all Processors supported can be found [here](pipeline.md#processors). |
| logging_level | `str` | `'INFO'` | Controls the level of logging information to display when the Pipeline is instantiated and run. Can be one of `'DEBUG'`, `'INFO'`, `'WARN'`, `'ERROR'`, `'CIRTICAL'`, or `'FATAL'`. Less and less information will be displayed from `'DEBUG'` to `'FATAL'`. |
| verbose | `str` | `None` | Simplified option for logging level. If `True`, logging level will be set to `'INFO'`. If `False`, logging level will be set to `'ERROR'`.  |
| use_gpu | `bool` | `True` | Attempt to use a GPU if available. Set this to `False` if you are in a GPU-enabled environment but want to explicitly keep Stanza from using the GPU. |
| kwargs | - | - | Options for each of the individual processors. See the individual processor pages for descriptions. |
| {processor}_model_path | - | - | Path to load an alternate model.  For example, `pos_model_path=xyz.pt` to load `xyz.pt` for the pos processor. |
| {processor}_pretrain_path | - | - | For processors which use word vectors, path to load an alternate set of word vectors.  For example, `pos_pretrain_path=abc.pt` to load the `abc.pt` pretrain for the pos processor. Will not work for NER, which has the vectors saved with the model. |

## Processors

Processors are units of the neural pipeline that perform specific NLP functions and create different annotations for a [`Document`](data_objects.md#document). The neural pipeline supports the following processors:

| Name | Processor class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| tokenize | Tokenize&shy;Processor | - | Segments a [`Document`](data_objects.md#document) into [`Sentence`](data_objects.md#sentence)s, each containing a list of [`Token`](data_objects.md#token)s. This processor also predicts which tokens are multi-word tokens, but leaves expanding them to the [MWTProcessor](mwt.md). | Tokenizes the text and performs sentence segmentation. |
| mwt | MWT&shy;Processor | tokenize | Expands multi-word tokens (MWTs) into multiple words when they are predicted by the tokenizer. Each [`Token`](data_objects.md#token) will correspond to one or more [`Word`](data_objects.md#word)s after tokenization and MWT expansion. | Expands [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html) predicted by the [TokenizeProcessor](tokenize.md). This is only applicable to some languages. |
| pos | POS&shy;Processor | tokenize, mwt | UPOS, XPOS, and UFeats annotations are accessible through [`Word`](data_objects.md#word)'s properties `pos`, `xpos`, and `ufeats`. | Labels tokens with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html). |
| lemma | Lemma&shy;Processor | tokenize, mwt, pos | Perform [lemmatization](https://en.wikipedia.org/wiki/Lemmatisation) on a [`Word`](data_objects.md#word) using the `Word.text` and `Word.upos` values. The result can be accessed as `Word.lemma`. | Generates the word lemmas for all words in the Document. |
| depparse | Depparse&shy;Processor | tokenize, mwt, pos, lemma | Determines the syntactic head of each word in a sentence and the dependency relation between the two words that are accessible through [`Word`](data_objects.md#word)'s `head` and `deprel` attributes. | Provides an accurate syntactic dependency parsing analysis. |
| ner | NER&shy;Processor | tokenize, mwt | Named entities accessible through [`Document`](data_objects.md#document) or [`Sentence`](data_objects.md#sentence)'s properties `entities` or `ents`. Token-level NER tags accessible through [`Token`](data_objects.md#token)'s properties `ner`. | Recognize named entities for all token spans in the corpus. |
| sentiment | Sentiment&shy;Processor | tokenize, mwt | Sentiment scores of 0, 1, or 2 (negative, neutral, positive).  Accessible using a  [`Sentence`](data_objects.md#sentence)'s `sentiment` property. | Assign per-sentence sentiment scores. |
| constituency | Constituency&shy;Processor | tokenize, mwt, pos | Parse trees accessible a  [`Sentence`](data_objects.md#sentence)'s `constituency`  property. | Parse each sentence in a document using a phrase structure parser. |

### Processor variants

New in v1.1
{: .label .label-green }

Sometimes you might want to build your own models to perform the tasks that existing processors already handle in the Stanza neural pipeline, or simply experiment with alternative toolkits for a specific task. Processor variants are here to help with that.

One example use case is using your own tokenizer for tokenization. Previously we have added support for popular tokenizers like spaCy for English and jieba for Chinese, and now we have made it much easier to add your own. You simply need to implement a `ProcessorVariant` and register it with Stanza using the `@register_processor_variant` decorator. For instance, this is how our spaCy tokenizer is implemented

```python
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant

@register_processor_variant('tokenize', 'spacy')
class SpacyTokenizer(ProcessorVariant):
    def __init__(self, config):
        # initialize spacy

    def process(self, text):
        # tokenize text with spacy
```

This allows the user to set `tokenize_with_spacy` as `True` (or `processors={"tokenize": "spacy"}`) when instantiating the pipeline to use it. In the case of the tokenizer, the `TokenizeProcessor` handles options such as pre-tokenization and pre-sengmentation, and only passes text to the variants when tokenization from raw text is needed.

Alternatively, one can also implement a processor variant as a drop-in replacement for a processor by setting `OVERRIDE` as `True` in the `ProcessorVariant` class, for instance

```python
@register_processor_variant("lemma", "cool")
class CoolLemmatizer(ProcessorVariant):
    ''' An alternative lemmatizer that lemmatizes every word to "cool". '''

    OVERRIDE = True

    def __init__(self, lang):
        pass

    def process(self, document):
        for sentence in document.sentences:
            for word in sentence.words:
                word.lemma = "cool"

        return document
```

This lemmatizer will replace all of the functionality of the Stanza lemmatizer when it's used in the pipeline, and lemmatize every single word to "cool".

{% include alerts.html %}
{{ note }}
{{ "It is essential to import the file where the variant is defined to trigger the `@register_processor_variant`" | markdownify }}
{{ end }}

### Building your own Processors and using them in the neural pipeline

New in v1.1
{: .label .label-green }

If you're looking to implement annotation capabilities that don't currently exist in Stanza and want to use it in the neural pipeline, it hasn't been easier. You can simply implement a `Processor` class and register it with Stanza using the `@register_processor` decorator, and then it's easy to use it in your project, and/or publish it for other Stanza users to use.

Here is an example:

```python
@register_processor("lowercase")
class LowercaseProcessor(Processor):
    ''' Processor that lowercases all text '''
    _requires = set(['tokenize'])
    _provides = set(['lowercase'])

    def __init__(self, config, pipeline, use_gpu):
        pass

    def _set_up_model(self, *args):
        pass

    def process(self, doc):
        doc.text = doc.text.lower()
        for sent in doc.sentences:
            for tok in sent.tokens:
                tok.text = tok.text.lower()

            for word in sent.words:
                word.text = word.text.lower()

        return doc
```

Once registered, you can use this processor in the pipeline as if it were one of Stanza's standard processors

```python
nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, lang='en', processors='tokenize,lowercase')
```

and in this case the processor will lowercase all text in the document.

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

You can find more examples about how to use these options [here](getting_started.md#building-a-pipeline).

{% include alerts.html %} {{ note }} {{ "To maximize speed performance, it is essential to run the pipeline on batches of documents. Running a for loop on one sentence at a time will be very slow. The best approach at this time is to concatenate documents together, with each document separated by a blank line (i.e., two line breaks \n\n). The tokenizer will recognize blank lines as sentence breaks. We are actively working on improving multi-document processing." | markdownify }}  {{ end }}

### Build Pipeline from a Config Dictionary

When there are many options you want to configure, or even set programmatically, it might not be convenient to set them one by one using keyword arguments to instantiate the Pipeline. In these cases, alternatively, you can build the desired pipeline with a config dictionary, allowing maximum customization for the pipeline:

```python
import stanza

config = {
        # Comma-separated list of processors to use
	'processors': 'tokenize,mwt,pos',
        # Language code for the language to build the Pipeline in
        'lang': 'fr',
        # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        # You only need model paths if you have a specific model outside of stanza_resources
	'tokenize_model_path': './fr_gsd_models/fr_gsd_tokenizer.pt',
	'mwt_model_path': './fr_gsd_models/fr_gsd_mwt_expander.pt',
	'pos_model_path': './fr_gsd_models/fr_gsd_tagger.pt',
	'pos_pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt',
        # Use pretokenized text as input and disable tokenization
	'tokenize_pretokenized': True
}
nlp = stanza.Pipeline(**config) # Initialize the pipeline using a configuration dict
doc = nlp("Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie .") # Run the pipeline on the pretokenized input text
print(doc) # Look at the result
```

Here, we can specify the language, processors, and paths for many Processor models all at once, and pass that to the Pipeline initializer. Note that config dictionaries and keyword arguments can be combined as well, to maximize your flexibility in using Stanza's neural pipeline.

### Processing Multiple Documents

New in v1.2
{: .label .label-green }

If you are annotating multiple pieces of text with Stanza, you might find it helpful to pass them all into the neural pipeline at once, and get back a list of documents on the output end. To achieve this, you can simply pass a list of stanza Documents into the pipeline for processing. For instance,

```python
import stanza
nlp = stanza.Pipeline(lang="en") # Initialize the default English pipeline
documents = ["This is a test document.", "I wrote another document for fun."] # Documents that we are going to process
in_docs = [stanza.Document([], text=d) for d in documents] # Wrap each document with a stanza.Document object
out_docs = nlp(in_docs) # Call the neural pipeline on this list of documents
print(out_docs[1]) # The output is also a list of stanza.Document objects, each output corresponding to an input Document object
```

You might find this useful when you have a large number of documents to process, but you would prefer Stanza to respect document boundaries. In future releases, we will optimize this interface behind the scenes to further speed up batched processing, and this would be significantly more efficient than processing your documents one by one.

The same usage pattern also applies to other Pipeline use cases, e.g., partially annotated documents (that are already in `stanza.Document` objects), pretokenized text (should be assigned to the `text` argument similarly to the raw text example above), etc.