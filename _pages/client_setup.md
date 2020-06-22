---
layout: page
title: Client Installation and Setup
keywords: CoreNLP, client, setup
permalink: '/client_setup.html'
nav_order: 1
parent: Stanford CoreNLP Client
---

To use the CoreNLP client, please first make sure that you have correctly installed the Stanza library. Follow [these instructions](installation_usage#installation) to install the library.

After the library is installed, you'll need to install the CoreNLP software package and make sure Stanza knows where the downloaded package is located on your computer. You have two ways of doing this.

## Automatic Installation

New in v1.1
{: .label .label-green }

If you just need to use the default CoreNLP library without additional models, the automatic installation function is the simplest way to do so:
```python
import stanza
stanza.install_corenlp(dir="YOUR_CORENLP_FOLDER", set_corenlp_home=True)
```
The first argument `dir` sets where you want your CoreNLP to be installed; by default it is installed in the `~/stanza_corenlp` folder, or if the environment variable `$STANZA_CORENLP_DIR` is set, it'll use the path specified in that variable.

The second argument `set_corenlp_home` specifies whether you want the `$CORENLP_HOME` environment variable to point to your installation directory. The `$CORENLP_HOME` variable is required by the Stanza client functionality, so we recommend you to do so. Alternatively you can manually set this by following the intructions below.

## Manual Installation

You can manually install CoreNLP if you need more models or more languages than the default package, or if the automatic method fails. You need to:

1. Download the latest version of Stanford CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/download.html) and follow the instructions to setup the environment.
2. Download model files for the language you want to annotate from [here](https://stanfordnlp.github.io/CoreNLP/download.html) and store them in the extracted CoreNLP folder. You can skip this step if you only want to use the default English models shipped with the CoreNLP software.
3. Set the `CORENLP_HOME` environment variable to the location of the CoreNLP root folder.  Example: `export CORENLP_HOME=/path/to/stanford-corenlp-full-2020-04-20`. Stanza will use this environment variable to locate the CoreNLP package at runtime.