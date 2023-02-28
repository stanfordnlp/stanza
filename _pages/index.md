---
layout: default
title: Overview
keywords: Stanza, Python, NLP, Natural Language Processing, Deep Learning, PyTorch
type: first_page
permalink: '/index.html'
nav_order: 1
homepage: true
---

# Stanza -- A Python NLP Package for Many Human Languages
{: .no_toc }
<img alt="PyPI Version" src="https://img.shields.io/pypi/v/stanza.svg?colorB=bc4545&style=flat-square" /> <img alt="Conda Versions" src="https://img.shields.io/conda/vn/stanfordnlp/stanza?color=bc4545&label=conda&style=flat-square" /> <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/stanza.svg?colorB=bc4545&style=flat-square" />

Stanza is a collection of accurate and efficient tools for the linguistic analysis of many human languages. Starting from raw text to syntactic analysis and entity recognition, Stanza brings state-of-the-art NLP models to languages of your choosing.
{: .fs-5 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

* TOC
{:toc}

<hr>

## About


Stanza is a Python natural language analysis package. It contains tools, which can be used in a pipeline, to convert a string containing human language text into lists of sentences and words, to generate base forms of those words, their parts of speech and morphological features, to give a syntactic structure dependency parse, and to recognize named entities. The toolkit is designed to be parallel among more than 70 languages, using the [Universal Dependencies formalism](https://universaldependencies.org).

Stanza is built with highly accurate neural network components that also enable efficient training and evaluation with your own annotated data. The modules are built on top of the [PyTorch](https://pytorch.org/) library. You will get much faster performance if you run the software on a GPU-enabled machine.

In addition, Stanza includes a Python interface to the [CoreNLP Java package](https://stanfordnlp.github.io/CoreNLP) and inherits additional functionality from there, such as constituency parsing, coreference resolution, and linguistic pattern matching.

To summarize, Stanza features:

* Native Python implementation requiring minimal efforts to set up;
* Full neural network pipeline for robust text analytics, including tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging, dependency parsing, and named entity recognition;
* Pretrained neural models supporting [70 (human) languages](models.md#human-languages-supported-by-stanza);
* A stable, officially maintained Python interface to CoreNLP.

Below is an overview of Stanza's neural network NLP pipeline:

<p align="center">
   <img src="assets/images/pipeline.png" style="width:90%">
</p>

## Getting Started

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

For more details on how to use the neural network pipeline, please see our [Installation](installation_usage.md), [Getting Started Guide](getting_started.md), and [Tutorials](tutorials.md) pages.

Aside from the neural pipeline, Stanza also provides the official Python wrapper for accessing the Java Stanford CoreNLP package. For more details, please see [Stanford CoreNLP Client](corenlp_client.md).

{% include alerts.html %}
{{ note }}
{{ "If you run into issues or bugs during installation or when you run Stanza, please check out [the FAQ page](faq.md). If you cannot find your issue there, please report it to us via [GitHub Issues](https://github.com/stanfordnlp/stanza/issues). A GitHub issue is also appropriate for asking general questions about using Stanza - please search the closed issues first!" | markdownify }}
{{ end }}

## License

Stanza is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (the "License"); you may not use the software package except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Main Contributors

The PyTorch implementation of Stanza's neural pipeline is due to [Peng Qi](http://qipeng.me), [Yuhao Zhang](http://yuhao.im), and [Yuhui Zhang](https://cs.stanford.edu/~yuhuiz/), with help from [Jason Bolton](mailto:jebolton@stanford.edu), [Tim Dozat](https://web.stanford.edu/~tdozat/) and [John Bauer](https://www.linkedin.com/in/john-bauer-b3883b60/). [John Bauer](https://www.linkedin.com/in/john-bauer-b3883b60/) currently leads the maintenance of this package.

The CoreNLP client is mostly written by [Arun Chaganty](http://arun.chagantys.org/), and [Jason Bolton](mailto:jebolton@stanford.edu) spearheaded merging the two projects together.

We are also grateful to community contributors for their help in improving Stanza.

## Citing Stanza in papers

If you use Stanza in your work, please cite this paper:

> Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. [Stanza: A Python Natural Language Processing Toolkit for Many Human Languages.](https://arxiv.org/abs/2003.07082) In Association for Computational Linguistics (ACL) System Demonstrations. 2020. \[[pdf](https://nlp.stanford.edu/pubs/qi2020stanza.pdf)\]\[[bib](https://nlp.stanford.edu/pubs/qi2020stanza.bib)\]
{: .citation }

If you use the biomedical and clinical model packages in Stanza, please also cite our JAMIA biomedical models paper:

> Yuhao Zhang, Yuhui Zhang, Peng Qi, Christopher D. Manning, Curtis P. Langlotz. [Biomedical and Clinical English Model Packages in the Stanza Python NLP Library](https://doi.org/10.1093/jamia/ocab090), Journal of the American Medical Informatics Association. 2021.
{: .citation }

If you use Stanford CoreNLP through the Stanza python client, please also follow the instructions [here](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers) to cite the proper publications.

## Links

[<i class="fab fa-github"></i> GitHub](https://github.com/stanfordnlp/stanza){: .btn .mr-2}
[<i class="fas fa-desktop"></i> Online Demo](http://stanza.run/){: .btn .mr-2}
[<i class="fab fa-python"></i> PyPI](https://pypi.org/project/stanza/){: .btn .mr-2}
[<i class="fas fa-toolbox"></i> CoreNLP](https://stanfordnlp.github.io/CoreNLP/){: .btn .mr-2}
[<i class="fas fa-user-friends"></i> Stanford NLP Group](https://nlp.stanford.edu/){: .btn .mr-2}
