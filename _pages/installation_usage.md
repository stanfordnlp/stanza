---
layout: page
title: Installation
keywords: installation-download
permalink: '/installation_usage.html'
nav_order: 1
parent: Usage
---

To use Stanza for text analysis, a first step is to install the package and download the models for the languages you want to analyze. After the download is done, an NLP pipeline can be constructed, which can process input documents and create annotations.

On this page, we introduce the installation of Stanza. For an introduction on how to use the neural pipeline, please see [Getting Started](getting_started.md).  For more examples of the neural pipeline, please check out our [Tutorials](tutorials). For usage information of the Stanford CoreNLP Python interface, please refer to the [CoreNLP Client](corenlp_client) page.

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
