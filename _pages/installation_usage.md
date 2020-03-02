---
title: Installation & Getting Started
keywords: installation-download
permalink: '/installation_usage.html'
---

To use StanfordNLP, you first need to install the package and download the model for the language you want to use. Then you can build the pipeline with downloaded models. Once the pipeline is built, you can process the document and get annotations.

## Installation

StanfordNLP supports Python 3.6 or later. We strongly recommend that you install StanfordNLP from PyPI. This should also help resolve all of the dependencies of StanfordNLP, for instance [PyTorch](https://pytorch.org/) 1.0.0 or above. If you already have [pip installed](https://pip.pypa.io/en/stable/installing/), simply run:
```bash
pip install stanfordnlp
```

If you currently have a previous version of `stanfordnlp` installed, use:
```bash
pip install stanfordnlp -U
```

Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of StanfordNLP and training your own models. For this option, run:
```bash
git clone https://github.com/stanfordnlp/stanfordnlp.git
cd stanfordnlp
pip install -e .
```

## Model Downloading

StanfordNLP provides simple, flexible, unified interfaces for downloading various models and building desired pipelines. A full list of available arguments can be found [here](pipeline.md#Options). Here we provide some intuitive examples covering most use cases:

Download the default pipeline for English:
```python
>>> stanfordnlp.download('en')
```

Download the default `tokenizer` and `pos tagger` for Chinese:
```python
>>> stanfordnlp.download('zh', processors='tokenize,pos')
```

Download the `tokenizer` and `multi-word expander` trained on GSD dataset for German:
```python
>>> stanfordnlp.download('de', processors='tokenize,mwt', package='gsd')
```

Download the `ner tagger` trained on CoNLL03 dataset and all other default processors for Dutch:
```python
>>> stanfordnlp.download('nl', processors={'ner': 'conll03'})
```

Download the `ner tagger` trained on WikiNER dataset, and other processors trained on PADT dataset for Arabic:
```python
>>> stanfordnlp.download('ar', processors={'ner': 'wikiner'}, package='padt')
```

Download the `tokenizer` trained on GSD dataset, `pos tagger` trained on Spoken dataset, `ner tagger` trained on CoNLL03 dataset, and default `lemmatizer` for French:
```python
>>> stanfordnlp.download('fr', processors={'tokenize': 'gsd', 'pos': 'spoken', 'ner': 'conll03', 'lemma': 'default'}, package=None)
```

Other arguments include specify model downloading directory and control how much information to print. 

Download the default pipeline for English to current working directory, and print all the information for debugging:
```python
>>> stanfordnlp.download('en', dir='.', logging_level='DEBUG')
```

## Pipeline loading

Besides exactly the same interfaces for loading as downloading, additional arguments can be found [here](pipeline.md#options), allowing users to control devices (cpu or gpu), use pretokenized text, disable sentence split, specify model path, etc. 

Load the default `tokenizer` for English:
```python
>>> nlp = stanfordnlp.download('en')
```

## Document Processing

Once the pipeline is loaded, you can simply pass the text to the pipeline and get the annotated document.

```python
>>> doc = nlp('Barack Obama was born in Hawaii.')
```

## Explore Annotations

After all processors are run, a [Document](data_objects#document) instance will be returned, which stores all annotation results. Within a Document, annotations are further stored in [Sentence](data_objects#sentence)s, [Token](data_object#token)s, [Word](data_objects#word)s, [Span](data_objects#span)s in a top-down fashion.

Print the text and POS tag of each word in the document:
```python
for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text, word.pos)
```

Print all entities and dependencies in the document:
```python
for sentence in doc.sentences:
    print(sentence.entities)
    print(sentence.dependencies)
```
