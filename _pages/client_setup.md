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

After Stanza is installed, you need to install the CoreNLP software package and make sure Stanza knows where the downloaded package is located on your computer. There are two ways to do this.

## Installing Java

The client is written in Java, so you will first need to install the
Java client.  CoreNLP requires Java 1.8 or higher.  You will also need
to add Java to your path.  On some systems, such as with the Windows
installer, this may happen automatically, or you may need to manually
update your path.

## Automated Installation

New in v1.1
{: .label .label-green }

The automated installation function is the simplest way to install CoreNLP along with the default models. You do not need to download and install CoreNLP yourself; rather Stanza does it for you. Most simply you can use:
```python
import stanza
stanza.install_corenlp()
```
By default, CoreNLP is installed in the `~/stanza_corenlp` folder.

Alternatively, if the `$CORENLP_HOME` environment variable is set, by default, CoreNLP is installed in the path specified by the `$CORENLP_HOME` environment variable. The install function can have an argument `dir` which specifies where you want your CoreNLP to be installed:
```python
import stanza
stanza.install_corenlp(dir="YOUR_CORENLP_FOLDER")
```
If you choose to set `dir` to your own customized directory, you also need to point the environment variable `$CORENLP_HOME` to this location after installing CoreNLP. On Linux/Unix or macOS, this can be done by running `export CORENLP_HOME=YOUR_CORENLP_FOLDER`.

Apart from the default package distribution, CoreNLP also provides [additional models for six different languages](https://stanfordnlp.github.io/CoreNLP/index.html#download). To install these additional models, you can do:
```python
stanza.download_corenlp_models(model='french', version='4.2.2', dir="YOUR_CORENLP_FOLDER")
```
Here the `model` argument specifies the model package that you want to install, and can be set to one of `'arabic', 'chinese', 'english', 'english-kbp', 'french', 'german', 'spanish'`; the `version` argument specifies the model version, for which `4.2.2` was the latest in 2021; and `dir` is needed to point to a customized CoreNLP installation location, or the models will be installed to the default location.


## Manual Installation

You can also manually install CoreNLP. You need to:

1. Download the latest version of Stanford CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/download.html) and follow the instructions to unpack it and set up the environment.
2. Download model files for the language you want to annotate from [here](https://stanfordnlp.github.io/CoreNLP/download.html) and store them in the extracted CoreNLP folder. You can skip this step if you only want to use the default English models shipped with the CoreNLP software.
3. Set the `CORENLP_HOME` environment variable to the location of the CoreNLP root folder.  Example: `export CORENLP_HOME=/path/to/stanford-corenlp-4.2.2`. Stanza will use this environment variable to locate the CoreNLP package at runtime.
