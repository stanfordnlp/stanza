---
layout: page
title: Client Installation and Setup
keywords: CoreNLP, client, setup
permalink: '/client_setup.html'
nav_order: 1
parent: Stanford CoreNLP Client
toc: false
---

To use the CoreNLP client, please first make sure that you have correctly installed the Stanza library. Follow [these instructions](installation_usage#installation) to install the library.

After the library is installed, you'll need to install the CoreNLP software package and make sure Stanza knows where the downloaded package is located on your computer. You need to:

<!-- There are two ways to do this. -->

<!-- ## Automated Installation

New in v1.1
{: .label .label-green }

The automated installation function is the simplest way to install CoreNLP along with the default models:
```python
import stanza
stanza.install_corenlp(dir="YOUR_CORENLP_FOLDER")
```
The first argument `dir` sets where you want your CoreNLP to be installed; or if `dir` is not set, by default it is installed in the path specified by the `$CORENLP_HOME` environment variable, or in the `~/stanza_corenlp` folder if `$CORENLP_HOME` isn't set either. If you choose to set `dir` to your own customized directory, you'll also need to point the environment variable `$CORENLP_HOME` to this location after installing CoreNLP. On Linux/Unix or macOS, this can be done with running `export CORENLP_HOME=YOUR_CORENLP_FOLDER`.

Apart from the default package distribution, CoreNLP also provides [additional models for six different languages](https://stanfordnlp.github.io/CoreNLP/index.html#download). To install these additional models, you can do:
```python
stanza.download_corenlp_models(model='french', version='4.0.0', dir="YOUR_CORENLP_FOLDER")
```
Here the `model` argument specifies the model package that you want to install, and can be set to one of `'arabic', 'chinese', 'english', 'english-kbp', 'french', 'german', 'spanish'`; the `version` argument specifies the model version, for which `4.0.0` is the latest; and `dir` needs to point to your customized CoreNLP installation location, or the models will be installed to the default location.


## Manual Installation

You can manually install CoreNLP if the automated method fails. You need to: -->

1. Download the latest version of Stanford CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/download.html) and follow the instructions to setup the environment.
2. Download model files for the language you want to annotate from [here](https://stanfordnlp.github.io/CoreNLP/download.html) and store them in the extracted CoreNLP folder. You can skip this step if you only want to use the default English models shipped with the CoreNLP software.
3. Set the `CORENLP_HOME` environment variable to the location of the CoreNLP root folder.  Example: `export CORENLP_HOME=/path/to/stanford-corenlp-full-2020-04-20`. Stanza will use this environment variable to locate the CoreNLP package at runtime.