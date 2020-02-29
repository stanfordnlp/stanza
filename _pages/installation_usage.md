---
title: Installation & Model Download
keywords: installation-download
permalink: '/installation_usage.html'
---

## Getting started

### Installation

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

### Model Downloading

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

### Pipeline loading

Besides exactly the same interfaces for loading as downloading, additional arguments can be found [here](pipeline.md#Options), allowing users to control devices (cpu or gpu), use pretokenized text, disable sentence split, specify model path, etc. 

Load the default `tokenizer` for English:
```python
>>> nlp = stanfordnlp.download('en', processors='tokenize')
```

### Document Processing




### Explore Annotations


## Quick Example

To try out StanfordNLP, you can simply follow these steps in the interactive Python interpreter:

```python
>>> import stanfordnlp
>>> stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
>>> nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

The last command here will print out the words in the first sentence in the input string (or `Document`, as it is represented in StanfordNLP), as well as the indices for the word that governs it in the Universal Dependencies parse of that sentence (its "head"), along with the dependency relation between the words. The output should look like:

```
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')
```

To build a pipeline for other languages, simply pass in the language code to the constructor like this `stanfordnlp.Pipeline(lang="fr")`. For a full list of languages (and their corresponnding language codes) supported by StanfordNLP, please see [this section](#human-languages-supported-by-stanfordnlp).

We also provide a [demo script](https://github.com/stanfordnlp/stanfordnlp/blob/master/demo/pipeline_demo.py) in our Github repostory that demonstrates how one uses StanfordNLP in other languages than English, for example Chinese (traditional)

```python
python demo/pipeline_demo.py -l zh
```

And expect outputs like the following:

```
---
tokens of first sentence:
達沃斯	達沃斯	PROPN
世界	世界	NOUN
經濟	經濟	NOUN
論壇	論壇	NOUN
是	是	AUX
每年	每年	DET
全球	全球	NOUN
政	政	PART
商界	商界	NOUN
領袖	領袖	NOUN
聚	聚	VERB
在	在	VERB
一起	一起	NOUN
的	的	PART
年度	年度	NOUN
盛事	盛事	NOUN
。	。	PUNCT

---
dependency parse of first sentence:
('達沃斯', '4', 'nmod')
('世界', '4', 'nmod')
('經濟', '4', 'nmod')
('論壇', '16', 'nsubj')
('是', '16', 'cop')
('每年', '10', 'nmod')
('全球', '10', 'nmod')
('政', '9', 'case:pref')
('商界', '10', 'nmod')
('領袖', '11', 'nsubj')
('聚', '16', 'acl:relcl')
('在', '11', 'mark')
('一起', '11', 'obj')
('的', '11', 'mark:relcl')
('年度', '16', 'nmod')
('盛事', '0', 'root')
('。', '16', 'punct')
```


