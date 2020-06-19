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

After the library is installed, you'll need to download the CoreNLP software package and make sure Stanza knows where the downloaded package is located on your computer. You need to:
1. Download the latest version of Stanford CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/download.html).
2. Download model files for the language you want to annotate from [here](https://stanfordnlp.github.io/CoreNLP/download.html) and store them in the extracted CoreNLP folder. You can skip this step if you only want to use the default English models shipped with the CoreNLP software.
3. Set the `CORENLP_HOME` environment variable to the location of the CoreNLP root folder.  Example: `export CORENLP_HOME=/path/to/stanford-corenlp-full-2020-04-20`. Stanza will use this environment variable to locate the CoreNLP package at runtime.