---
layout: page
title: Adding a new CharLM model
keywords: charlm, stanza, model training
permalink: '/new_language_charlm.html'
nav_order: 5
parent: Training
---

## Introduction

For adding a new languages, we provide scripts to automate large parts of the process.  Scripts for converting raw text to conllu and conllu to a charlm dataset can be found in [stanza/utils/charlm/conll17_to_text.py](https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/charlm/conll17_to_text.py) and [stanza/utils/charlm/make_lm_data.py](https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/charlm/make_lm_data.py)

* Gather a ton of tokenized text.  Ideally gigabytes.  Wikipedia is a good place to start for raw text, but in that case you will need to tokenize it.
  * One such source of text is [the conll17 shared task](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1989)
  * Another possibility is to gather the Common Crawl data yourself, such as from [Oscar](https://oscar-corpus.com/), and download the Wikipedia dump for the language
  * There is a script in the dev branch, [`stanza.utils.charlm.dump_oscar`](https://github.com/stanfordnlp/stanza/blob/dev/stanza/utils/charlm/dump_oscar.py), which should help exporting Oscar data from HuggingFace to the charlm
* If the data you gathered was from the conll17 shared task, we provide a script to turn it into txt files.  Run ```python3 -m stanza.utils.charlm.conll17_to_text ~/extern_data/finnish/conll17/Finnish/```  This will convert conllu or conllu.xz files to txt and put them in the same directory.
* Run ```python3 -m stanza.utils.charlm.make_lm_data extern_data/charlm_raw extern_data/charlm```  This will convert text files in the `charlm_raw` directory to a suitable dataset in `extern_data/charlm`.  You may need to adjust your paths.
* Forward: ```python3 -m stanza.models.charlm --train_dir extern_data/charlm/fi/conll17/train --eval_file extern_data/charlm/fi/conll17/dev.txt.xz --direction forward --lang fi --shorthand fi_conll17  --mode train```
* Backward: ```python3 -m stanza.models.charlm --train_dir extern_data/charlm/fi/conll17/train --eval_file extern_data/charlm/fi/conll17/dev.txt.xz --direction backward --lang fi --shorthand fi_conll17  --mode train```
* This will take days or weeks to fully train.

For most languages, the current defaults are sufficient, but for some languages the learning rate is too aggressive and leads to NaNs in the training process.  For example, for Finnish, we used the following parameters: `--lr0 10`

## Step by Step Training

First, we need a large amount of text data.  For this model, we choose
two sources: Oscar Common Crawl and Wikipedia.

There is a script to copy Oscar from HuggingFace:

```bash
python3 -m stanza.utils.charlm.dump_oscar bn --output /nlp/scr/horatio/oscar/
```

{% include alerts.html %}
{{ note }}
{{ "To use this script, you will need to install the HuggingFace library `datasets`." | markdownify }}
{{ end }}

We also download Wikipedia from the
[Wikipedia dumps archive](https://dumps.wikimedia.org/backup-index-bydb.html).
If a dump exists for your language, it will be under the language code
for that language.
We will use Prof. Attardi's
[WikiExtractor](https://github.com/attardi/wikiextractor) tool to
remove the markup, and it works on the `latest-pages-meta-current`
file, so that is what we download.

```bash
wget https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-meta-current.xml.bz2
```

You can then use the WikiExtractor to extract the text from the
Wikipedia dump you just downloaded:

```bash
python -m wikiextractor.WikiExtractor bnwiki-latest-pages-meta-current.xml.bz2
```

This splits the text into multiple subdirectories full of small files
`AA, AB, ...` depending on the size.  The splits are smaller than we
need, but we can combine them:

```bash
for i in `ls text`; do echo $i; cat text/$i/* > $i.txt; xz $i.txt; done
```

We now have an Oscar dump and a Wikipedia dump.  We can turn this raw
data into train/dev/test splits for the charlm.  First, we organize
the raw data into one directory.  Then, we run the `make_lm_data` script.
On our cluster, we put all of our raw charlm data into
`/u/nlp/software/stanza/charlm_raw`
and the train/dev/test splits into `/u/nlp/software/stanza/charlm`
You can choose different base paths, of course.

```bash
export CHARLM_DIR=/u/nlp/software/stanza/charlm
export CHARLM_RAW_DIR=/u/nlp/software/stanza/charlm_raw
# move the Oscar & Wikipedia .xz files to this directory
mkdir -p $CHARLM_RAW_DIR/bn/oscar

ls $CHARLM_RAW_DIR/bn/oscar
AA.txt.xz  oscar_dump_000.txt.xz  oscar_dump_007.txt.xz  oscar_dump_014.txt.xz  oscar_dump_021.txt.xz
AB.txt.xz  oscar_dump_001.txt.xz  oscar_dump_008.txt.xz  oscar_dump_015.txt.xz  oscar_dump_022.txt.xz
AC.txt.xz  oscar_dump_002.txt.xz  oscar_dump_009.txt.xz  oscar_dump_016.txt.xz  oscar_dump_023.txt.xz
AD.txt.xz  oscar_dump_003.txt.xz  oscar_dump_010.txt.xz  oscar_dump_017.txt.xz
AE.txt.xz  oscar_dump_004.txt.xz  oscar_dump_011.txt.xz  oscar_dump_018.txt.xz
AF.txt.xz  oscar_dump_005.txt.xz  oscar_dump_012.txt.xz  oscar_dump_019.txt.xz
AG.txt.xz  oscar_dump_006.txt.xz  oscar_dump_013.txt.xz  oscar_dump_020.txt.xz

python3 -m stanza.utils.charlm.make_lm_data $CHARLM_RAW_DIR $CHARLM_DIR --langs bn --packages oscar
```

{% include alerts.html %}
{{ note }}
{{ "make_lm_data has several subprocess calls which are not expected to work on Windows." | markdownify }}
{{ end }}

{% include alerts.html %}
{{ note }}
{{ "Please double check that the directory with the data is `$CHARLM_RAW_DIR/<lang>/<dataset>`" | markdownify }}
{{ end }}

You can now run the charlm.  This will take days.  Remember to update the language!

```bash
python3 -m stanza.models.charlm --train_dir $CHARLM_DIR/bn/oscar/train --eval_file $CHARLM_DIR/bn/oscar/dev.txt.xz --direction forward --lang bn --shorthand bn_oscar --mode train > bn_forward.out 2>&1 &
python3 -m stanza.models.charlm --train_dir $CHARLM_DIR/bn/oscar/train --eval_file $CHARLM_DIR/bn/oscar/dev.txt.xz --direction backward --lang bn --shorthand bn_oscar --mode train > bn_backward.out 2>&1 &
```

You can tell when the model has converged and is no longer improving by looking for the eval scores:

```bash
grep "eval checkpoint" bn_*.out
```

Alternatively, you can tie it in to wandb (requires Stanza 1.4.1 or later) by signing in to wandb and adding `wandb_name` to the original command line:

```bash
python3 -m stanza.models.charlm --train_dir $CHARLM_DIR/bn/oscar/train --eval_file $CHARLM_DIR/bn/oscar/dev.txt.xz --direction forward --lang bn --shorthand bn_oscar --mode train --wandb_name bn_oscar_forward_charlm > bn_forward.out 2>&1 &
python3 -m stanza.models.charlm --train_dir $CHARLM_DIR/bn/oscar/train --eval_file $CHARLM_DIR/bn/oscar/dev.txt.xz --direction backward --lang bn --shorthand bn_oscar --mode train --wandb_name bn_oscar_backward_charlm > bn_backward.out 2>&1 &
```

Once it has converged satisfactorily, you can copy the models to the
expected locations in your stanza resources and rerun the NER.  If you
follow the name structure used in this example command line,
`run_ner.py` will look for and find the charlm in these exact paths.
Remember that you can update $STANZA_RESOURCES_DIR if you need.

```bash
mkdir -p ~/stanza_resources/bn/forward_charlm
cp saved_models/charlm/bn_oscar_forward_charlm.pt ~/stanza_resources/bn/forward_charlm/oscar.pt

mkdir -p ~/stanza_resources/bn/backward_charlm
cp saved_models/charlm/bn_oscar_backward_charlm.pt ~/stanza_resources/bn/backward_charlm/oscar.pt

python3 -m stanza.utils.training.run_ner bn_daffodil --charlm oscar --save_name bn_daffodil_charlm.pt
```

## Integrating with other models

Once the charlm is trained, you can integrate it to NER as follows.
Other models which support charlm are similar.

```bash
mkdir -p ~/stanza_resources/bn/forward_charlm
cp saved_models/charlm/bn_oscar_forward_charlm.pt ~/stanza_resources/bn/forward_charlm/oscar.pt

mkdir -p ~/stanza_resources/bn/backward_charlm
cp saved_models/charlm/bn_oscar_backward_charlm.pt ~/stanza_resources/bn/backward_charlm/oscar.pt

python3 -m stanza.utils.training.run_ner bn_daffodil --charlm oscar --save_name bn_daffodil_charlm.pt
```

