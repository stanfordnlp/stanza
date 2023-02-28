---
layout: page
title: Download Models
keywords: download models
permalink: '/download_models.html'
nav_order: 4
parent: Usage
toc: false
---

Stanza provides pretrained NLP models for a total 70 human languages. On this page we provide detailed information on how to download these models to process text in a language of your choosing.

Pretrained models in Stanza can be divided into two categories, based on the datasets they were trained on:
1. Universal Dependencies (UD) models, which are trained on the UD treebanks, and cover functionalities including tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging and dependency parsing;
2. NER models, which support named entity tagging for 8 languages, and are trained on various NER datasets.

{% include alerts.html %}
{{ note }}
{{ "For more information on what models are available for download, please see [Available Models](available_models.md)." | markdownify }}
{{ end }}


Downloading Stanza models is as simple as calling the `stanza.download()` method. We provide detailed examples on how to use the `download` interface on the [Getting Started](getting_started.md#building-a-pipeline) page. Detailed descriptions of all available options (i.e., arguments) of the `download` method are listed below:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| lang | `str` | `'en'` | Language code (e.g., `"en"`) or language name (e.g., `"English"`) for the language to process with the Pipeline. See the tables of available models below for a complete list of supported languages. |
| model_dir | `str` | `'~/stanza_resources'` | Directory for storing the models downloaded for Stanza. By default, Stanza stores its models in a folder in your home directory.  |
| package | `str` | `'default'` | Package to download for processors, where each package typically specifies what data the models are trained on. We provide a "default" package for all languages that contains NLP models most users will find useful, which will be used when the `package` argument isn't specified. See table below for a complete list of available packages. |
| processors | `dict`&nbsp;or&nbsp;`str` | `dict()` | [Processor](pipeline.md#processors)s to download models for. This can either be specified as a comma-seperated list of processor names to use (e.g., `'tokenize,pos'`), or a Python dictionary with processor names as keys and package names as corresponding values (e.g., `{'tokenize': 'ewt', 'pos': 'ewt'}`). All unspecified processors will fall back to using the package specified by the `package` argument. A list of all Processors supported can be found [here](pipeline.md#processors).   |
| logging_level | `str` | `'INFO'` | Controls the level of logging information to display during download. Can be one of `'DEBUG'`, `'INFO'`, `'WARN'`, `'ERROR'`, `'CIRTICAL'`, or `'FATAL'`. Less information will be displayed going from `'DEBUG'` to `'FATAL'`. |
| verbose | `str` | `None` | Simplified option for logging level. If `True`, logging level will be set to `'INFO'`. If `False`, logging level will be set to `'ERROR'` (i.e., only show errors).  |

{% include alerts.html %}
{{ note }}
{{ "You can override the default location `~/stanza_resources` by setting an environmental variable called `STANZA_RESOURCES_DIR`." | markdownify }}
{{ end }}

{% include alerts.html %}
{{ note }}
{{ "Offline installation of models is possible by copying the models from an existing `~/stanza_resources` installation.  This may be useful for docker installation or machines with no internet connection." | markdownify }}
{{ end }}

{% include alerts.html %}
{{ note }}
{{ "As of v1.4.0, creating a `Pipeline` will automatically try to download missing models.  You can turn off this behavior with `download_method=None` when creating a pipeline." | markdownify }}
{{ end }}

