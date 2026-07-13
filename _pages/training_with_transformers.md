---
layout: page
title: Training with Transformer Embeddings
keywords: stanza, model training, bert, transformer, peft
permalink: '/training_with_transformers.html'
nav_order: 1.5
parent: Training
---

## Overview

In addition to the word vector and charlm embeddings described in [Model Training and Evaluation](training_and_evaluation.md), the POS/morphological features tagger, dependency parser, NER tagger, constituency parser, and sentiment classifier can all be trained using contextual embeddings from a transformer model.

Stanza's pretrained models each use a transformer we have found to work best for that language and task. However, you are not limited to those choices — any transformer model on [Hugging Face](https://huggingface.co/models) can be used when retraining a model, whether that's a different existing model or one you have finetuned yourself.

## Using the Default Transformer: `--use_bert`

Each of the following training scripts accepts a `--use_bert` flag:

```bash
python -m stanza.utils.training.run_pos ${corpus} --use_bert
python -m stanza.utils.training.run_depparse ${corpus} --use_bert
python -m stanza.utils.training.run_ner ${corpus} --use_bert
python -m stanza.utils.training.run_constituency ${corpus} --use_bert
python -m stanza.utils.training.run_sentiment ${corpus} --use_bert
```

Adding `--use_bert` will train the model using the transformer we currently consider the best choice for that language, based on our own internal benchmarking. This is the same transformer used by the corresponding pretrained model we distribute for that language, if one exists.

## Choosing a Specific Transformer: `--bert_model`

If you want to use a different transformer than our current default — for example, a newer model, a domain-specific model, or one you have finetuned yourself — you can specify it directly with `--bert_model`:

```bash
python -m stanza.utils.training.run_pos ${corpus} --bert_model ${transformer_name}
```

`${transformer_name}` can be any model identifier recognized by Hugging Face's `transformers` library (i.e., anything you could pass to `AutoModel.from_pretrained`). This includes models hosted on the Hugging Face Hub as well as models saved locally on disk.

`--bert_model` implies `--use_bert`, so there is no need to specify both flags together.

{% include alerts.html %}
{{ note }}
{{ "Not all transformer architectures behave identically, and some may require more GPU memory or different batch sizes than our defaults. You may need to adjust `--batch_size` or other training arguments accordingly." | markdownify }}
{{ end }}

### Finding the Current Default for a Language

The transformer used by `--use_bert` for each language is defined in [`stanza/resources/default_packages.py`](https://github.com/stanfordnlp/stanza/blob/main/stanza/resources/default_packages.py):

- `TRANSFORMERS` is a map from language code to the transformer we currently select for that language. Each entry also includes a comment explaining the reasoning behind that choice (e.g., benchmark results, coverage, size tradeoffs), so it's a good place to check before deciding whether to override it with `--bert_model`.
- `TRANSFORMER_NICKNAMES` provides short, easier-to-type aliases for some of the longer Hugging Face model names used in `TRANSFORMERS`.

If you want to try one of our existing choices for a *different* language than its default, or want to see the full name behind a nickname, these two maps are the place to look.

## PEFT Finetuning

Stanza also integrates with [PEFT](https://github.com/huggingface/peft) (Parameter-Efficient Fine-Tuning) to allow finetuning the transformer itself alongside the rest of the model, rather than only using it as a fixed feature extractor. This is available for the same set of modules: POS, depparse, NER, constituency, and sentiment.

In our own experiments, PEFT finetuning has mainly been beneficial for the **constituency parser** and the **sentiment classifier**. For POS, depparse, and NER, we have not observed a consistent improvement from enabling PEFT, so it's worth treating it as experimental for those modules and comparing against a non-PEFT baseline on your own data before relying on it.

For details on the specific PEFT-related flags for each module, refer to the arguments listed in that module's training script (`run_${module}.py`).

## Results

Results with transformer models coming soon.
