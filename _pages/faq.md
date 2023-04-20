---
layout: page
title: Frequently Asked Questions (FAQ)
keywords: stanza, frequently asked questions, faq
permalink: '/faq.html'
nav_order: 1
parent: Resources
---

## Model Output

### Model predictions are wrong on some of my examples, is this normal?

This is absolutely normal, as all models in Stanza (yes, even tokenization!) are statistical. Although they are quite accurate, it does not mean these models are perfect. Therefore, it's quite likely that you'll find cases where the model prediction clearly doesn't make sense, but statistically speaking, it shouldn't be too far off on a large collection of text from the [performance we report](performance.md) as long as the genre of your text is similar to what the models are trained on.

### The model prediction is inconsistent between Stanza and CoreNLP, different versions of Stanza, or their online demos

Stanza's neural pipeline use fundamentally different models from CoreNLP for all tasks, and are usually trained on different data, so it is not unexpected that their behaviors will differ.

As for online demos, it is possible that some demos are using models that are different from the latest models available for download, so it is not impossible that there are slight differences there as well.

### Can I use Stanza models in CoreNLP, or the other way around?

Since Stanza's neural pipeline use fundamentally different models from CoreNLP for all tasks, it will not be possible to use Stanza's model in CoreNLP or the other way around.

However, you could use CoreNLP for part of the annotation (e.g., tokenization) through the [`CoreNLPClient`](corenlp_client.md), and use the resulting annotations as input to Stanza's neural pipeline.

### Can I run POS tagging/morphological feature tagging/lemmatization/dependency parsing without expanding multi-word tokens (MWTs)?

For syntactic tasks such as POS/morphological feature tagging, lemmatization, and dependency parsing, Stanza uses data made available through the [Universal Dependencies](https://universaldependencies.org/) project which makes the distinction between tokens (substrings of the input text) and syntactic words (see the [UD documentation on this](https://universaldependencies.org/u/overview/tokenization.html) for more information). This means if the language/dataset you want to use was deemed to contain multi-word tokens (MWTs), unfortunately nothing beyond tokenization and sentence segmentation can happen unless MWTs are expanded with the [MWT expansion model](mwt.md) in the pipeline (with the exception of [named entity recognition](ner.md), which are based on tokens!).

## Troubleshooting Download & Installation

### Getting `ERROR: Could not find a version` `that satisfies the requirement torch` when installing Stanza

This is usually because PyTorch doesn't have a version that Stanza requires for install through `pip`. You can usually work around this issue by installing PyTorch from your package manager (e.g., Anaconda) first, before trying to install Stanza.

### Getting `module 'stanza' has no attribute 'download'` when downloading models with Stanza

This is likely because you're using Python 2. Note that Stanza only supports Python 3.6 or later.

### Model download is very slow or I cannot connect to the server

Although we try our best to keep our model server available, it does become unavailable from time to time due to various reasons, e.g., hardware updates, power outages, etc. These will usually be resolved within a few hours. Please be patient while we fix issues on our side!

### Getting `requests.exceptions.ConnectionError` when downloading models

This is an known issue for users from some certain areas, such as China. A common reason for this is that a connection to the `raw.githubusercontent.com` URL cannot be established, and therefore the resource file required for downloading models cannot be accessed. Users have widely reported that using a VPN that provides stable access to GitHub services can solve this issue.

### Stanza is trying to download models and/or its resources file when offline.  How can I stop that?

When building a `Pipeline`, Stanza will try to download the resources file to see if anything has changed, then download any updated models.  You can turn this behavior off by creating the `Pipeline` with the `download_method=None` parameter.  You can make it reuse an existing resources file but try to download any updated models with `download_method="reuse_resources"`.

## Troubleshooting Running Stanza

### Why do I keep getting a `SyntaxError: invalid syntax` error message while trying to `import stanza`?

Stanza will not work with Python 3.5 or below. If you have trouble importing the package, please try to upgrade your Python.

### Getting `Segmentation fault` or other uninterpretable non-Python errors when trying to run the neural pipeline

This is ususally caused by a corrupted installation of PyTorch in your environment. Try reinstalling PyTorch and Stanza.

### Why am I getting an `OSError: [Errno 22] Invalid argument` error and therefore a `Vector file is not provided` exception while the model is being loaded?

If you are getting this error, it is very likely that you are running macOS and using Python with version <= 3.6.7 or <= 3.7.1. If this is the case, then you are affected by a [known Python bug](https://bugs.python.org/issue24658) on macOS, and upgrading your Python to >= 3.6.8 or >= 3.7.2 should solve this issue.

If you are not running macOS or already have the specified Python version and still seeing this issue, please report this to us via the [GitHub issue tracker](https://github.com/stanfordnlp/stanza/issues).

