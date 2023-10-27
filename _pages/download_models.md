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

## Automatic download

Pretrained models in Stanza can be divided into four categories, based on the datasets they were trained on:
- Universal Dependencies (UD) models, which are trained on the UD treebanks, and cover functionalities including tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging and dependency parsing;
- NER models, which support named entity tagging for 8 languages, and are trained on various NER datasets.
- Constituency models, trained on a specific constituency parser dataset
- Sentiment models, similarly trained on a dataset for that specific language and task

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


## Manual download

In some cases, it is not convenient, practical, or possible to
download models automatically.  In such cases, it will be necessary to
manually download the models and populate the resources directory.

By default, the resources are downloaded to `~/stanza_resources`.
There are a couple ways to change this.  One is by defining the
environment variable `$STANZA_RESOURCES_DIR`.  The other is, when
creating a `Pipeline`, specify a different directory via the
`model_dir` parameter.

If you are in a situation where you cannot download resources while
creating the `Pipeline`, it will also be necessary to provide the
argument `download_method=None`, as by default the `Pipeline` checks
for updated models at creation time.

Once you have chosen the location for the models download,
if you are able to download models programmatically, the easiest way
to populate that directory will be using the `stanza.download()` method described above.

Otherwise, models are available for download in separate HuggingFace
repos for the language in question.  Each repo is specified by the
short language code used by Stanza

So, the models for English are [in the stanza-en repo](https://huggingface.co/stanfordnlp/stanza-en)

You will also need to download the resources to download the
resources.json file appropriate for the Stanza version you are using.
This is available [in a Stanford git repo](https://github.com/stanfordnlp/stanza-resources)
The resources file goes in `$STANZA_RESOURCES/resources.json` (without the version number)

Each language has the default models packaged in a `default.zip` file.
The English one, for example,
[is in this subdirectory of the English models tree](https://huggingface.co/stanfordnlp/stanza-en/tree/main/models)
This goes in a language specific directory of `$STANZA_RESOURCES_DIR`
so for example the English package goes in `$STANZA_RESOURCES_DIR/en/default.zip`
From there, unzip the models package.

As of version 1.6.1, downloading `resources.json`, downloading the
`default.zip` file for English, and unzipping `default.zip` results in
the following:

```
$STANZA_RESOURCES_DIR/
$STANZA_RESOURCES_DIR/resources.json
$STANZA_RESOURCES_DIR/en
$STANZA_RESOURCES_DIR/en/default.zip
$STANZA_RESOURCES_DIR/en/backward_charlm/1billion.pt
$STANZA_RESOURCES_DIR/en/constituency/ptb3-revised_charlm.pt
$STANZA_RESOURCES_DIR/en/depparse/combined_charlm.pt
$STANZA_RESOURCES_DIR/en/forward_charlm/1billion.pt
$STANZA_RESOURCES_DIR/en/lemma/combined_nocharlm.pt
$STANZA_RESOURCES_DIR/en/ner/ontonotes_charlm.pt
$STANZA_RESOURCES_DIR/en/pos/combined_charlm.pt
$STANZA_RESOURCES_DIR/en/pretrain/conll17.pt
$STANZA_RESOURCES_DIR/en/pretrain/fasttextcrawl.pt
$STANZA_RESOURCES_DIR/en/sentiment/sstplus.pt
$STANZA_RESOURCES_DIR/en/tokenize/combined.pt
```

This is the expected layout if manually downloading model files.

To manually download a different package for the same language,
individual model files can be downloaded and added to the appropriate
place.  For example, to use the `ncbi_disease.pt` NER model for
English, that can be downloaded from
`https://huggingface.co/stanfordnlp/stanza-en/tree/main/models/ner`
into `STANZA_RESOURCES_DIR/en/ner/ncbi_disease.pt`.  You can explore
the stanza-en tree to find the specific model you are looking for.

If downloading several individual models, we are aware that can be
tedious, especially when checking for updates.  Please let us know if
you need a package similar to `default.zip` for a different
combination of models.
