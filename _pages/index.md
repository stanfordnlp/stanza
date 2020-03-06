---
title: Stanza Python Library
keywords: Stanza, Python, NLP, Natural Language Processing, Deep Learning, PyTorch
type: first_page
permalink: '/index.html'
homepage: true
---

## About

Stanza is a Python natural language analysis package. It contains tools, which can be used in a pipeline, to convert a string containing human language text into lists of sentences and words, to generate base forms of those words, their parts of speech and morphological features, to give a syntactic structure dependency parse, and to recognize named entities, which is designed to be parallel among more than 70 languages, using the [Universal Dependencies formalism](https://universaldependencies.org). 

This package is built with highly accurate neural network components that enable efficient training and evaluation with your own annotated data. The modules are built on top of [PyTorch](https://pytorch.org/). You will get much faster performance if you run this system on a GPU-enabled machine.

![Stanza Pipeline](images/pipeline.png)

In addition, it is able to call the [CoreNLP Java package](https://stanfordnlp.github.io/CoreNLP) and inherits additonal functionality from there, such as constituency parsing, coreference resolution, and linguistic pattern matching.

Stanza features:

* Native Python implementation requiring minimal efforts to set up;
* Full neural network pipeline for robust text analytics, including tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging, dependency parsing, and named entity recognition;
* Pretrained neural models supporting [66 (human) languages](models.md#human-languages-supported-by-stanza);
* A stable, officially maintained Python interface to CoreNLP.

## Get Started

We strongly recommend installing Stanza with `pip`, which is as simple as:

```bash
pip install stanza
```

To see Stanza's neural pipeline in action, you can launch the Python interactive interpreter, and try the following commands:

```python
>>> import stanza
>>> stanza.download('en') # download English model
>>> nlp = stanza.Pipeline('en') # initialize English neural pipeline
>>> doc = nlp("Barack Obama was born in Hawaii.") # run annotation over a sentence
```

You should be able to see all the annotations in the example by running the following commands: 

```python
>>> print(doc)
>>> print(doc.entities)
```

For more details, please see our [getting started guide](installation_usage.md) and [tutorial](tutorial.md).

Aside from the neural pipeline, Stanza also provides the official Python wrapper for acessing the Java Stanford CoreNLP Server. For more details, please see [Stanford CoreNLP Client](corenlp_client.md).

**Note**: If you run into issues during installation or when you run the example scripts, please check out [this FAQ page](faq.md). If you cannot find your issue there, please report it to us on GitHub.


## License

Stanza is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (the "License"), you may not use the software package except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Citing Stanza in papers

If you use the Stanza neural pipeline in your work, please cite this paper:

> Peng Qi, Timothy Dozat, Yuhao Zhang and Christopher D. Manning. 2018. [Universal Dependency Parsing from Scratch](https://nlp.stanford.edu/pubs/qi2018universal.pdf) In *Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies*, pp. 160-170. \[[pdf](https://nlp.stanford.edu/pubs/qi2018universal.pdf)\] \[[bib](https://nlp.stanford.edu/pubs/qi2018universal.bib)\]

If you use Stanford CoreNLP through the Stanza python client, please follow the instructions [here](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers) to cite the proper publications.

## Links

* [GitHub](https://github.com/stanfordnlp/stanza)
* [PyPI](https://pypi.org/project/stanza/)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
