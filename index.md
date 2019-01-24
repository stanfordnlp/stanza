---
title: About
keywords: overview, about
type: first_page
homepage: true
---

## About

StanfordNLP is the combination of the software package used by the Stanford team in the CoNLL 2018 Shared Task on Universal Dependency Parsing, and the group's official Python interface to the [Stanford CoreNLP software](https://stanfordnlp.github.io/CoreNLP). Aside from the functions it inherits from CoreNLP, it contains tools to convert a string of text to lists of sentences and words, generate base forms of those words, their parts of speech and morphological features, and a syntactic structure that is designed to be parallel among more than 70 languages.

This package is built with highly accurate neural network components that enables efficient training and evaluation with your own annotated data. The modules are built on top of [PyTorch](https://pytorch.org/).

Choose StanfordNLP if you need:

* An integrated NLP toolkit with a broad range of grammatical analysis tools
* A fast, robust annotator for arbitrary texts
* A modern, regularly updated package, with cutting edge tools for text analytics
* Support for a wide range of (human) languages
* A stable, officially maintained Python interface to CoreNLP

## Installation

We strongly recommend installing StanfordNLP with `pip`, which is as simple as

```bash
pip install stanfordnlp
```

To see StanfordNLP's neural pipeline in action, you can launch the Python interactive interpreter, and try the following commands

```python
>>> import stanfordnlp
>>> stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
>>> nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

At the end, you should be able to see the dependency parse of the first sentence in the example. For more details, please see our [getting started guide](/installation_download.html#getting-started).

Aside from the neural pipeline, StanfordNLP also provides the official Python wrapper for acessing the Java Stanford CoreNLP Server. To use it, you first need to set up the CoreNLP package as follows

* Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and models for the language you wish to use.
* Put the model jars in the distribution folder
* Tell the python code where Stanford CoreNLP is located: `export CORENLP_HOME=/path/to/stanford-corenlp-full-2018-10-05`

After CoreNLP is set up, you can follow our [demo script](https://github.com/stanfordnlp/stanfordnlp/blob/master/demo/corenlp.py) to test it out.

## License

StanfordNLP is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (the "License"), you may not use the software package except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Citing StanfordNLP in papers

If you use the StanfordNLP neural pipeline in your work, please cite this paper:

> Peng Qi, Timothy Dozat, Yuhao Zhang and Christopher D. Manning. 2018. [Universal Dependency Parsing from Scratch](https://nlp.stanford.edu/pubs/qi2018universal.pdf) In *Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies*, pp. 160-170. \[[pdf](https://nlp.stanford.edu/pubs/qi2018universal.pdf)\] \[[bib](https://nlp.stanford.edu/pubs/qi2018universal.bib)\]

If you use Stanford CoreNLP through the StanfordNLP python client, please follow the instructions [here](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers) to cite the proper publications.