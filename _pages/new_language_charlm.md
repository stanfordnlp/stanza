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
  * Another possibility is to gather the Common Crawl data yourself, such as from [Community-OSCAR](https://huggingface.co/datasets/oscar-corpus/community-oscar), and download the Wikipedia dump for the language.  See the [Community-OSCAR section below](#gathering-community-oscar-data) for the scripts we provide to do this.
* If the data you gathered was from the conll17 shared task, we provide a script to turn it into txt files.  Run ```python3 -m stanza.utils.charlm.conll17_to_text ~/extern_data/finnish/conll17/Finnish/```  This will convert conllu or conllu.xz files to txt and put them in the same directory.
* Run ```python3 -m stanza.utils.charlm.make_lm_data extern_data/charlm_raw extern_data/charlm```  This will convert text files in the `charlm_raw` directory to a suitable dataset in `extern_data/charlm`.  You may need to adjust your paths.
* Forward: ```python3 -m stanza.models.charlm --train_dir extern_data/charlm/fi/conll17/train --eval_file extern_data/charlm/fi/conll17/dev.txt.xz --direction forward --shorthand fi_conll17  --mode train```
* Backward: ```python3 -m stanza.models.charlm --train_dir extern_data/charlm/fi/conll17/train --eval_file extern_data/charlm/fi/conll17/dev.txt.xz --direction backward --shorthand fi_conll17  --mode train```
* This will take days or weeks to fully train.

For most languages, the current defaults are sufficient, but for some languages the learning rate is too aggressive and leads to NaNs in the training process.  For example, for Finnish, we used the following parameters: `--lr0 10`

## Step by Step Training

First, we need a large amount of text data.  For this model, we choose
two sources: Community-OSCAR Common Crawl and Wikipedia.

```bash
export CHARLM_DIR=/u/nlp/software/stanza/charlm
export CHARLM_RAW_DIR=/u/nlp/software/stanza/charlm_raw
```

## Gathering Community-OSCAR data

[Community-OSCAR](https://huggingface.co/datasets/oscar-corpus/community-oscar)
is a gated dataset on HuggingFace.  You will need to accept the access
agreement on the dataset page and obtain a HuggingFace token before
using the scripts below.  Export your token as `HF_TOKEN` or pass it
via `--hf_token`.

We provide three scripts in `stanza/utils/charlm/` for working with
Community-OSCAR:

**`community_oscar_inventory.py`** lists the snapshots and languages
available in Community-OSCAR, which is useful for knowing which slices
exist before you start downloading.  It requires only one API call
regardless of how many snapshots are available, and caches the result
locally for 24 hours.

```bash
# List all snapshots containing Bengali
python3 stanza/utils/charlm/community_oscar_inventory.py --lang bn
```

This prints snapshots newest-first in `lang:snapshot` format, ready to
paste directly into the `--slices` argument of the dedup script:

```
bn:2024-38
bn:2024-33
bn:2024-30
...
```

**`community_oscar_dedup.py`** downloads one or more slices, deduplicates
across all of them (by exact URL and by MinHash near-duplicate detection),
and writes one plain-text output file per slice.  Deduplication state is
shared across slices in the order given, so the first-listed snapshot's
content takes priority.

```bash
pip install datasketch huggingface_hub zstandard
```

```bash
python3 stanza/utils/charlm/community_oscar_dedup.py \
    --slices bn:2024-38 bn:2024-33 bn:2024-30 \
    --output_dir /nlp/scr/horatio/oscar/bn \
    --minhash_threshold 0.7
```

{% include alerts.html %}
{{ note }}
{{ "The `--minhash_threshold` parameter controls how similar two documents must be (in terms of word-set overlap) to be considered near-duplicates. 0.7 (70% shared vocabulary) is a good default for most languages. Run `community_oscar_inspect_similarity.py` on your output to verify the threshold is well-calibrated for your language." | markdownify }}
{{ end }}

At the end of the run, the script prints a summary table showing how
many documents and words each snapshot contributed, plus the marginal
contribution of each slice — useful for deciding how many snapshots to
include.

**`community_oscar_inspect_similarity.py`** samples random document
pairs from an output file and reports the Jaccard and TLSH similarity
distributions.  Run this after deduplication to verify that the output
is genuinely diverse and the threshold you chose has a low false-positive
rate for your language.

```bash
pip install datasketch py-tlsh zstandard  # py-tlsh optional but recommended
```

```bash
python3 stanza/utils/charlm/community_oscar_inspect_similarity.py \
    /nlp/scr/horatio/oscar/bn/bn_2024-38.txt
```

The output files from the dedup script are plain `.txt` files (one
document per line).  Before passing them to `make_lm_data`, compress
them to `.xz` and move them to the appropriate raw data directory:

```bash
for f in /nlp/scr/horatio/oscar/bn/*.txt; do xz "$f"; done
mkdir -p $CHARLM_RAW_DIR/bn/oscar
mv /nlp/scr/horatio/oscar/bn/*.txt.xz $CHARLM_RAW_DIR/bn/oscar/
```

**Using OSCAR 2023 (legacy)**

An older script, `dump_oscar.py`, downloads from the OSCAR 2023 dataset
on HuggingFace rather than Community-OSCAR.  This path has two
significant limitations as of 2026: new user access to OSCAR 2023 is
no longer being granted, and the script requires a rollback to
`datasets` 3.x since it is incompatible with the current package
version.  If you already have access and are working in an older
environment, it can still be used:

```bash
pip install "datasets<4.0"
python3 -m stanza.utils.charlm.dump_oscar bn --output /nlp/scr/horatio/oscar/
```

This produces output files named `oscar_dump_000.txt.xz`,
`oscar_dump_001.txt.xz`, etc., which can be passed to `make_lm_data`
the same way as the Community-OSCAR files above.

## Wikipedia Downloads

We also download Wikipedia from the
[Wikipedia dumps archive](https://dumps.wikimedia.org/backup-index-bydb.html).
If a dump exists for your language, it will be under the language code
for that language.
We will use Prof. Attardi's
[WikiExtractor](https://github.com/attardi/wikiextractor) tool to
remove the markup, and it works on the `latest-pages-meta-current`
file, so that is what we download.

Until a couple of fixes get released upstream (a template
self-inclusion loop that could hang indefinitely on certain articles,
plus a few output-cleanliness improvements), install from our fork
instead of the upstream package:

```bash
pip install git+https://github.com/AngledLuffa/wikiextractor.git
```

```bash
wget https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-meta-current.xml.bz2
```

You can then use the WikiExtractor to extract the text from the
Wikipedia dump you just downloaded. We use `--text` to skip the
`<doc>...</doc>` XML wrapper (which is just extraction metadata, not
Wikipedia content, and would otherwise pollute the LM training data)
and `--discard_empty` to skip near-empty documents (redirects,
disambiguation stubs, etc.) that don't contribute any useful text:

```bash
python -m wikiextractor.WikiExtractor --text --discard_empty bnwiki-latest-pages-meta-current.xml.bz2
```

This splits the text into multiple subdirectories full of small files
`AA, AB, ...` depending on the size.  The splits are smaller than we
need, but we can combine them:

```bash
for i in `ls text`; do echo $i; cat text/$i/* > $i.txt; xz $i.txt; done
```

With `--text --discard_empty`, these files are already plain text with
no `<doc>` wrapper tags and no near-empty stub documents, so this
concatenation step needs no further cleanup.

We now have a Community-OSCAR dump and a Wikipedia dump.  We can turn this raw
data into train/dev/test splits for the charlm.  First, we organize
the raw data into one directory.  Then, we run the `make_lm_data` script.
On our cluster, we put all of our raw charlm data into
`/u/nlp/software/stanza/charlm_raw`
and the train/dev/test splits into `/u/nlp/software/stanza/charlm`
You can choose different base paths, of course.

```bash
# move the Oscar & Wikipedia .xz files to this directory
mkdir -p $CHARLM_RAW_DIR/bn/oscar

ls $CHARLM_RAW_DIR/bn/oscar
AA.txt.xz  bn_2024-30.txt.xz  bn_2024-38.txt.xz
AB.txt.xz  bn_2024-33.txt.xz

python3 -m stanza.utils.charlm.make_lm_data $CHARLM_RAW_DIR $CHARLM_DIR --langs bn --packages oscar
```

{% include alerts.html %}
{{ note }}
{{ "make_lm_data has several subprocess calls which are not expected to work on Windows.  This will be fixed in the next release of Stanza!" | markdownify }}
{{ end }}

{% include alerts.html %}
{{ note }}
{{ "Please double check that the directory with the data is `$CHARLM_RAW_DIR/<lang>/<dataset>`" | markdownify }}
{{ end }}

You can now run the charlm.  This will take days.  Remember to update the language!

```bash
python3 -m stanza.models.charlm --train_dir $CHARLM_DIR/bn/oscar/train --eval_file $CHARLM_DIR/bn/oscar/dev.txt.xz --direction forward --shorthand bn_oscar --mode train > bn_forward.out 2>&1 &
python3 -m stanza.models.charlm --train_dir $CHARLM_DIR/bn/oscar/train --eval_file $CHARLM_DIR/bn/oscar/dev.txt.xz --direction backward --shorthand bn_oscar --mode train > bn_backward.out 2>&1 &
```

You can tell when the model has converged and is no longer improving by looking for the eval scores:

```bash
grep "eval checkpoint" bn_*.out
```

Alternatively, you can tie it in to wandb (requires Stanza 1.4.1 or later) by signing in to wandb and adding `wandb_name` to the original command line:

```bash
python3 -m stanza.models.charlm --train_dir $CHARLM_DIR/bn/oscar/train --eval_file $CHARLM_DIR/bn/oscar/dev.txt.xz --direction forward --shorthand bn_oscar --mode train --wandb_name bn_oscar_forward_charlm > bn_forward.out 2>&1 &
python3 -m stanza.models.charlm --train_dir $CHARLM_DIR/bn/oscar/train --eval_file $CHARLM_DIR/bn/oscar/dev.txt.xz --direction backward --shorthand bn_oscar --mode train --wandb_name bn_oscar_backward_charlm > bn_backward.out 2>&1 &
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

## Citation

> Akbik, Alan and Blythe, Duncan and Vollgraf, Roland.  [Contextual String Embeddings for Sequence Labeling](https://aclanthology.org/C18-1139), Proceedings of the 27th International Conference on Computational Linguistics, Association for Computational Linguistics, 2018.
{: .citation }
