---
layout: page
title: Language Identification
keywords: language identification
permalink: '/langid.html'
nav_order: 12
parent: Neural Pipeline
---

## Overview

With Stanza a user can detect the language of text and route texts of different languages to different language specific pipelines. The current distributed model is a character level Bi-LSTM trained off of text snippets from the UD 2.5 dataset. The model works on a variety of text types, including short text snippets (10 chars), sentences, tweets, and paragraphs. 

Currently the model detects the following languages:

```python
af ar be bg bxr ca cop cs cu da de el en es et eu fa fi fr fro ga gd gl got grc he hi hr hsb hu hy id it ja kk kmr ko la lt lv lzh mr mt nl nn no olo orv pl pt ro ru sk sl sme sr sv swl ta te tr ug uk ur vi wo zh-hans zh-hant
```

## Get The Language ID Model And Resources For English and French

```python
import stanza

stanza.download(lang="multilingual")
stanza.download(lang="en")
stanza.download(lang="fr")
```

## Basic Language ID Example

With the langid processor, one can identify the language of text. The detected language will be stored in the lang field of the Document.

```python
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline

nlp = Pipeline(lang="multilingual", processors="langid")
docs = ["Hello world.", "Bonjour le monde!"]
docs = [Document([], text=text) for text in docs]
nlp(docs)
print("\n".join(f"{doc.text}\t{doc.lang}" for doc in docs)) 
```

## Use A Custom Model
```python
nlp = Pipeline(lang="multilingual", processors="langid", langid_model_path="/path/to/model.pt")
```

## Apply Text Cleaning To Tweets

If running on tweets, it is helpful to clean the text before submitting to the model. The text cleaning will remove shortened urls, hashtags, user handles, and emojis. This is not turned on by default.

```python
nlp = Pipeline(lang="multilingual", processors="langid", langid_clean_text=True)
```

## Restricting Language Predictions To A Subset Of Languages

In some scenarios you may know that the possible language is only from a small subset of languages. The language id module can be configured to only predict from this subset. This example demonstrates restricting predictions to English or French.

```python
nlp = Pipeline(lang="multilingual", processors="langid", langid_lang_subset=["en","fr"])
```

If you are using the `MultilingualPipeline`, you can set this by adding `langid_lang_subset` to the `lang_id_config`:

```python
lang_id_config = {"langid_lang_subset": ['ar', 'hi']}
nlp = MultilingualPipeline(lang_id_config=lang_id_config)
```

## Basic Multilingual Pipeline Example

A `MultilingualPipeline` will detect the language of text, and run the appropriate language specific Stanza pipeline on the text. The `MultilingualPipeline` will maintain a cache of pipelines for each language. This example demonstrates handling some English and French text. Each example is classified as English or French, and then an appropriate English or French pipeline is run on the text.

```python
from stanza.pipeline.multilingual import MultilingualPipeline

nlp = MultilingualPipeline()
docs = ["Hello world!", "C'est une phrase française.", "This is an English sentence."]
docs = nlp(docs)
for doc in docs:
    print("---")
    print(f"text: {doc.text}")
    print(f"lang: {doc.lang}")
    print(f"{doc.sentences[0].dependencies_string()}")
```

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| model_dir | str | DEFAULT_MODEL_DIR | Where language id and language specific resources are stored. |
| lang_id_config | dict | None | Configurations to use for language identification. |
| lang_configs | dict | None | Mapping of language name --> pipeline configurations for that language |
| ld_batch_size | int | 64 | Batch size to use for language identification |
| max_cache_size | int | 10 | Max number of pipelines to cache |

## Configure Multilingual Pipeline

You can configure the language identification system and each language specific pipeline in the `MultilingualPipeline`. When the `MultilingualPipeline` is constructed, it can be fed a dictionary with one entry per language, where each entry is a dictionary with that language's settings. The `langid` processor itself can be configured as well with a separate dictionary with `langid` settings.

This example demonstrates activating the text cleaning for the language identification module and setting the cached English pipeline’s NER model. 

```python
from stanza.pipeline.multilingual import MultilingualPipeline

lang_id_config = {"langid_clean_text": True}
lang_configs = {"en": {"processors": {"ner": "conll03"}}}
nlp = MultilingualPipeline(lang_id_config=lang_id_config, lang_configs=lang_configs)
docs = ["Hello world.", "Bonjour le monde! #thisisfrench #ilovefrance"]
docs = nlp(docs)
for doc in docs:
    print("---")
    print(f"text: {doc.text}")
    print(f"lang: {doc.lang}")
    print(f"{doc.sentences[0].dependencies_string()}")
```

## Set Multilingual Pipeline Cache Size

A `MultilingualPipeline` keeps a cache of pipelines for each language. The maximum size of the cache can be configured at pipeline construction.

When a document is processed, its language is detected, and the appropriate pipeline is used. If the cache is at capacity and a new language is
detected, the least recently used language pipeline is removed and a new pipeline is added.

```python
nlp = MultilingualPipeline(max_cache_size=2)
```

## Training Your Own Model

You can train your own model with the `lang_identifier.py` script.

The data should be stored in a directory, with 3 files: `train.jsonl`, `dev.jsonl`, and `test.jsonl`. The data format is one entry per line, each entry is JSON specifying the text and the language label.

```python
{"text": "Hello world.", "label": "en"}
```

Training can be launched with the following command (assume the `*.jsonl` files are in a directory called `data`)

```bash
python -m stanza.models.lang_identifier --data-dir data  --eval-length 10 --randomize --save-name model.pt --num-epochs 100
```

This command will run training with the data in `train.jsonl` and evaluate with data in `dev.jsonl`.

When the `--randomize` option is used, snippets of between 5 and 20 characters are sampled from each training example and used as the final training examples for each epoch. So in one epoch the training example "This is an English sentence." might yield "This is" "an English", and "sentence.", and in another it might yield "This is an", "English sentence."

The length of the snippets can be set with `--randomize-lengths-range`.

To get the best performance on short strings (character length=10), it is crucial to train on relatively short examples.

`--eval-length` will determine the length of the examples used for validation

## Evaluate A Model

A trained model can be evaluated on any data set with the following command

```bash
python -m stanza.models.lang_identifier --data-dir data --load-model model.pt --mode eval --eval-length 50 --save-name model-results.jsonl
```

This command will look for the file `test.jsonl` in `data` and produce evaluation numbers for the data in that file.

The overall accuracy will be displayed, and a `.jsonl` file with various evaluation info including the accuracy, the confusion matrix, and per-language F1, precision, and recall will be produced.
