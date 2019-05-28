---
title: Installation & Model Download
keywords: installation-download
permalink: '/installation_usage.html'
---

## Getting started

### Installation

To get started with StanfordNLP, we strongly recommend that you install it through [PyPI](https://pypi.org/). Once you have [pip installed](https://pip.pypa.io/en/stable/installing/), simply run in your command line

```bash
pip install stanfordnlp
```

This will take care of all of the dependencies necessary to run StanfordNLP. The neural pipeline of StanfordNLP depends on PyTorch 1.0.0 or a later version with compatible APIs.

### Quick Example

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

### Troubleshooting

- Why do I keep getting a `SyntaxError: invalid syntax` error message while trying to import stanfordnlp?

   StanfordNLP will not work with Python 3.5 or below. If you have trouble importing the package, please try to upgrade your Python.

- Why am I getting an `OSError: [Errno 22] Invalid argument` error and therefore a `Vector file is not provided` exception while the model is being loaded?

   If you are getting this error, it is very likely that you are running macOS and using Python with version <= 3.6.7 or <= 3.7.1. If this is the case, then you are affected by a [known Python bug](https://bugs.python.org/issue24658) on macOS, and upgrading your Python to >= 3.6.8 or >= 3.7.2 should solve this issue. If you are not running macOS or already have the specified Python version and still seeing this issue, please report this to us via the [GitHub issue tracker](https://github.com/stanfordnlp/stanfordnlp/issues).

