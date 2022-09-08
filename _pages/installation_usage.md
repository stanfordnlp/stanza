---
layout: page
title: Installation & Getting Started
keywords: installation-download
permalink: '/installation_usage.html'
nav_order: 1
parent: Usage
---

To use Stanza for text analysis, a first step is to install the package and download the models for the languages you want to analyze. After the download is done, an NLP pipeline can be constructed, which can process input documents and create annotations.

On this page, we mainly introduce the installation of Stanza and simple examples for using its neural pipeline. For more examples of the neural pipeline, please check out our [Tutorials](tutorials). For usage information of the Stanford CoreNLP Python interface, please refer to the [CoreNLP Client](corenlp_client) page.

## Installation

### pip

Stanza supports Python 3.6 or later. We recommend that you install Stanza via [pip](https://pip.pypa.io/en/stable/installing/), the Python package manager. To install, simply run:
```bash
pip install stanza
```
This should also help resolve all of the dependencies of Stanza, for instance [PyTorch](https://pytorch.org/) 1.3.0 or above.

If you currently have a previous version of `stanza` installed, use:
```bash
pip install stanza -U
```

### Anaconda

To install Stanza via Anaconda, use the following conda command:

```bash
conda install -c stanfordnlp stanza
```

Note that for now installing Stanza via Anaconda does not work for Python 3.8. For Python 3.8 please use pip installation.

### From Source

Alternatively, you can also install from source via Stanza's git
repository, which will give you more flexibility in developing on top
of Stanza. For this option, first install
[Cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
(optional, but highly recommended), then install
[PyTorch](https://pytorch.org/), then run

```bash
git clone https://github.com/stanfordnlp/stanza.git
cd stanza
pip install -e .
```


## Building a Pipeline

Stanza provides simple, flexible, and unified interfaces for downloading and running various NLP models. At a high level, to start annotating text, you need to first initialize a [Pipeline](pipeline.md#pipeline), which pre-loads and chains up a series of [Processor](pipeline.md#processors)s, with each processor performing a specific NLP task (e.g., tokenization, dependency parsing, or named entity recognition).

Downloading models and building a pipeline of models shares roughly the same interface. Additionally, when building a pipeline, you can add customized options that control devices (CPU or GPU), allow pretokenized text, or specify model path, etc. Here we aim to provide examples that cover common use cases. For all available options in the download and pipeline interface, please refer to the [Downloading Models](models#downloading-and-using-models) and [Pipeline](pipeline.md#pipeline) pages.

<br />
The following minimal example will download and load default processors into a pipeline for English:
```python
>>> import stanza
>>> nlp = stanza.Pipeline('en')
```

### Specifying Processors

You can specify the processors to download or load, by listing the processor names in a comma-separated string. For example, here we only download and load the default `tokenize` ([TokenizeProcessor](tokenize.md)) and `pos` ([POSProcessor](pos.md)) processors for Chinese:
```python
nlp = stanza.Pipeline('zh', processors='tokenize,pos')
```

Note that the model of a processor has to be downloaded before it can be loaded into a pipeline.

{% include alerts.html %}
{{ note }}
{{ "You can check out all supported processors and their names in this [Processors List](pipeline.md#processors)." | markdownify }}
{{ end }}

### Specifying Model Packages

By default, all languages are shipped with a `default` package, which will be downloaded and loaded when no package name is specified. However, you can tell Stanza to download or load a specific package with the optional `package` option. For example, we can download and load the [TokenizeProcessor](tokenize.md) and [MWTProcessor](mwt.md) trained on the `GSD` dataset for German with:
```python
nlp = stanza.Pipeline('de', processors='tokenize,mwt', package='gsd')
```

In some cases, you may want to use a specific package for one processor, but remain `default` for the rest of the processors. This can be done with a dictionary-based `processors` argument. This example shows how to download and load the [NERProcessor](ner.md) trained on the Dutch `CoNLL02` dataset, but use `default` package for all other processors for Dutch:
```python
nlp = stanza.Pipeline('nl', processors={'ner': 'conll02'})
```

Similarly, the following example shows how to use the [NERProcessor](ner.md) trained on the `WikiNER` dataset, while use models trained on the `lassysmall` dataset for all other processors for Dutch:
```python
nlp = stanza.Pipeline('nl', processors={'ner': 'wikiner'}, package='lassysmall')
```

Rarely, you may want to have full control over package names for all processors, instead of relying on the `default` package at all. This can be enabled by setting `package=None`. The following example shows how to use a `GSD` [TokenizeProcessor](tokenize.md), a `HDT` [POSProcessor](pos.md), and a `CoNLL03` [NERProcessor](ner.md), and a `default` [LemmaProcessor](lemma.md) for German:
```python
processor_dict = {
    'tokenize': 'gsd', 
    'pos': 'hdt', 
    'ner': 'conll03', 
    'lemma': 'default'
}
nlp = stanza.Pipeline('de', processors=processor_dict, package=None)
```

{{ note }}
{{ "For the list of all available packages for different languages, please refer to the [Models](models.md) page." | markdownify }}
{{ end }}

### Downloading models for offline usage

In each of the examples above, it is possible to download the models ahead of time and request that the Pipeline not download anything.  `stanza.download()` will download individual models or entire packages using the same interface as `Pipeline`, and then the `Pipeline` has a flag to turn off downloads.

A couple examples:

```python
import stanza
nlp = stanza.Pipeline('en')
```

```python
import stanza
stanza.download('zh', processors='tokenize,pos')
nlp = stanza.Pipeline('zh', processors='tokenize,pos', download_method=None)
```

```python
processor_dict = {
    'tokenize': 'gsd', 
    'pos': 'hdt', 
    'ner': 'conll03', 
    'lemma': 'default'
}
stanza.download('de', processors=processor_dict, package=None)
nlp = stanza.Pipeline('de', processors=processor_dict, package=None)
```

There is also a mechanism for only attempting to download models when
a particular package is missing.  It will reuse an existing
`resources.json` file rather than trying to download it, though.

```
from stanza.pipeline.core import DownloadMethod
nlp = stanza.Pipeline('zh', processors='tokenize,pos', download_method=DownloadMethod.REUSE_RESOURCES)
```

This feature is new as of 1.4.0.  Prior versions required an initial call to `download` before building the `Pipeline`.

### Controlling Logging from the Pipeline

By default, the pipeline will print model loading info and processor-specific logs to the standard output stream. The level of logs printed can be specified with the `logging_level` argument. The following example shows how to download and load the English pipeline while printing only warnings and errors:
```python
stanza.download('en', logging_level='WARN')
nlp = stanza.Pipeline('en', logging_level='WARN')
```

The pipeline interface also allows the use of a `verbose` option to quickly suppress all non-error logs when running the pipeline:
```python
nlp = stanza.Pipeline('en', verbose=False)
```


### Controlling Devices

Stanza is implemented to be "CUDA-aware", meaning that it will run its processors on a CUDA-enabled GPU device whenever such a device is available, or otherwise CPU will be used. If processing a lot of text, we suggest that you run the pipeline on GPU devices for maximum speed, but Stanza also runs fine on CPU. You can force the pipeline to always run on CPU by setting `use_gpu=False` when initializing the pipeline:
```python
nlp = stanza.Pipeline('en', use_gpu=False)
```

## Annotating a Document

Annotating text is simple after a [Pipeline](pipeline.md#pipeline) is built and finishes loading: you can simply pass the text to the pipeline instance and access all annotations from the returned [Document](data_objects#document) object:

```python
doc = nlp('Barack Obama was born in Hawaii.')
```

Within a [Document](data_objects#document), annotations are further stored in [Sentence](data_objects#sentence)s, [Token](data_objects#token)s, [Word](data_objects#word)s in a top-down fashion. An additional [Span](data_objects#span) object may be used to store annotations such as named entity mentions. Here we provide some simple examples to manipulate the returned annotations.

The following example shows how to print the text, lemma and POS tag of each word in each sentence of an annotated document:
```python
for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text, word.lemma, word.pos)
```

The following example shows how to print all named entities and dependencies in a document:
```python
for sentence in doc.sentences:
    print(sentence.ents)
    print(sentence.dependencies)
```

{{ note }}
{{ "A list of all data objects and their attributes and methods can be found on the [Data Objects](data_objects#document) page." | markdownify }}
{{ end }}
