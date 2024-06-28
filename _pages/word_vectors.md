---
layout: page
title: Word Vectors
keywords: stanza, model training, embedding
permalink: '/word_vectors.html'
nav_order: 3
parent: Training
---

## Downloading Word Vectors

To replicate the system performance on the CoNLL 2018 shared task, we have prepared a script for you to download all word vector files. Simply run from the source directory:
```bash
bash scripts/download_vectors.sh ${wordvec_dir}
```
where `${wordvec_dir}` is the target directory to store the word vector files, and should be the same as where the environment variable `WORDVEC_DIR` is pointed to.

The above script will first download the pretrained word2vec embeddings released from the CoNLL 2017 Shared Task, which can be found [here](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989/word-embeddings-conll17.tar?sequence=9&isAllowed=y). For languages not in this list, it will download the [FastText embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) from Facebook. Note that the total size of all downloaded vector files will be ~30G, therefore please use this script with caution.

After running the script, your embedding vector files will be organized in the following way:
`${WORDVEC_DIR}/{language}/{language_code}.vectors.xz`. For example, the word2vec file for English should be put into `$WORDVEC_DIR/English/en.vectors.xz`. If you use your own vector files, please make sure you arrange them in a similar fashion as described above.

{% include alerts.html %}
{{ note }}
{{ "If you only want one language's word vectors, you can get them from your [STANZA_RESOURCES](download_models.md) directory.  For example, word vectors used for English go to `~stanza_resources/en/pretrain/combined.pt` by default" | markdownify }}
{{ end }}

Several of the vector files we use are from the [Wiki version of the
fasttext vectors](https://fasttext.cc/docs/en/crawl-vectors.html).  We
have started noting those with `fasttextwiki` in the distributions.

```
@inproceedings{grave2018learning,
  title={Learning Word Vectors for 157 Languages},
  author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```


```
@article{bojanowski2017enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  issn={2307-387X},
  pages={135--146}
}
```

## Using alternate word vectors

The simplest way to retrain models for an existing language with new data is to use the existing word vectors.  Generally we redistribute word vectors built with word2vec or fasttext.

If you retrain the models with new word vectors, you will need to provide the path for those word vectors when creating a pipeline.  Otherwise, the pipeline will try to use the default word vectors for that language and/or package.  To specify a different set of word vectors, you can supply the following arguments as relevant:

```
pos_pretrain_path
depparse_pretrain_path
sentiment_pretrain_path
```

This is in addition to specifying the path for the model you have retrained.  Therefore, the complete arguments for initializing a pipeline with a new POS model trained with new word vectors will look like this:

```python
pipe = stanza.Pipeline(lang="en", processors="tokenize,pos", pos_pretrain_path="new_en_wv.pt", pos_model_path="new_pos_model.pt")
```

Prior to 1.4.1, the NER models included the word vectors in the models
themselves, but this was wasteful in terms of disk space and memory.
As of 1.4.1, the NER models no longer include the word vectors.  Use
ner_model_path to specify the word vector path if needed.


The pretrain embedding file expected by the pipeline is the `.pt` format PyTorch uses to save models.  The module which loads embeddings will convert a text file to a `.pt` file if needed, so you can use the following code snippet to create the `.pt` file:

```python
from stanza.models.common.pretrain import Pretrain
pt = Pretrain("foo.pt", "new_vectors.txt")
pt.load()
```

## Utility Scripts

In general we convert the embeddings into a PyTorch module for faster
loading and smaller disk sizes.  A script is provided which does that:

```
python3 stanza/models/common/convert_pretrain.py ~/stanza/saved_models/pos/fo_fasttext.pretrain.pt ~/extern_data/wordvec/fasttext/faroese.txt -1
```

The third argument sets the limit on how many vectors to keep.  In
most cases you will not want to keep them all, as the resulting file
could be GB large with a long tail of junk that provides no value to
the model.  We typically use 150000 as a compromise between
completeness and size/memory.

New in v1.2.1
{: .label .label-green }

There is also a script for counting how many times words in a UD training set appear in an embedding:

```
stanza/models/common/count_pretrain_coverage.py
```

### Large vector files

In some cases, the word vector files are so large that they will not
fit in memory.  The conversion program does not compensate for that at all.
To handle this problem, you can use `head` to get just the lines we will make use of:

```bash
cd ~/extern_data/wordvec/fasttext   # or wherever you put the vectors
gunzip cc.bn.300.vec.gz
head -n 150001 cc.bn.300.vec > cc.bn.300.head   # +1 as many text formats include a header
python3 -m stanza.models.common.convert_pretrain ~/stanza_resources/bn/pretrain/fasttext.pt cc.bn.300.head 150000
```

## Word Vector Sources

In addition to the CoNLL shared task vectors and the Fasttext vectors
mentioned above, some languages use specific word vector packages.

### CoNLL 17

For many of the languages which were in the conll 17 shared task, the word vectors used by default are the word vectors provided in that task.

New in v1.5.1
{: .label .label-green }

Starting with Stanza 1.5.1, the conll17 word vectors are all marked as such in the pretrain filename.

### Fasttext

Other languages use either the [Fasttext for 157 languages](https://fasttext.cc/docs/en/crawl-vectors.html) or [Fasttext Wiki](https://fasttext.cc/docs/en/pretrained-vectors.html) word vectors.

New in v1.5.1
{: .label .label-green }

Starting with Stanza 1.5.1, those vectors are marked with either `fasttext157` or `fasttextwiki`, as appropriate.

### Ancient Hebrew

A work studying word embeddings on old languages covered Ancient
Hebrew.  Those vectors were slightly more effective than using modern
Hebrew vectors on the first official
[Ancient Hebrew UD dataset](https://github.com/UniversalDependencies/UD_Ancient_Hebrew-PTNK)

| Pretrain     |  POS dev  | POS test | depparse LAS dev | depparse LAS test |
| :------:     | :-------: | :----:   | :------------:   | :-------------:   |
| none         |  85.64    |  88.15   |  80.73           |  85.99            |
| conll17 HE   |  86.39    |  88.90   |  80.41           |  85.63            |
| utah         |  86.45    |  88.90   |  80.64           |  86.02            |

Note that with one author from Pomona, and one author from Claremont,
we split the difference and called this embedding the Utah embedding,
since that is where it is currently hosted.

[Evaluating Word Embeddings on Low-Resource Languages](https://aclanthology.org/2020.eval4nlp-1.17) (Stringham & Izbicki, Eval4NLP 2020)

### Armenian and Western Armenian

[Glove vectors specifically for Armenian](https://github.com/ispras-texterra/word-embeddings-eval-hy)
improved POS and depparse F1 by about 1.  They also had much better coverage of Western Armenian,
for which there is a separate UD dataset.

Avetisyan, Karen and Ghukasyan, Tsolak (2019).
[Word Embeddings for the Armenian Language: Intrinsic and Extrinsic Evaluation](https://arxiv.org/abs/1906.03134).

### Chinese (Simplified)

The NER model for Stanza for 1.5.0 or earlier used the
[fasttextwiki](https://fasttext.cc/docs/en/pretrained-vectors.html)
word vectors.  The other models, such as POS or depparse, used a
version of those vectors where any words that still had traditional
characters were remapped to simplified characters instead.

We observe that for each of several tasks, the Fasttext 157 vectors
perform better.  Therefore, starting with Stanza 1.5.1, we use that
vector file for each of the Chinese tasks.  Incidentally, this makes
the download somewhat smaller, as only one set of vectors is needed.

We also tested [VCWE](https://github.com/HSLCY/VCWE), but found that
the fasttext vectors performed better for these tasks.

| Pretrain     |  POS  | depparse | ner    | sentiment |
| :------:     | :---: | :----:   | :----: | :----:    |
| fasttext157  | 94.78 | 78.58    | 74.96  | 65.91     |
| fasttextwiki | 94.47 | 78.28    | 73.15  | 62.65     |
| vcwe         | 94.16 | 78.09    | 74.72  | 64.24     |

### Classical Armenian

The [word vectors](https://github.com/stanfordnlp/stanza/issues/1343)
for Classical Armenian are licensed under the
[CC BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)
for non-profit use only.  They were produced at the
[Caval project](https://www.phil.uni-wuerzburg.de/vgsp/forschung/projekte/),
which published
[Development of Linguistic Annotation Toolkit for Classical Armenian in SpaCy, Stanza, and UDPipe](https://github.com/caval-repository/xcl_nlp/blob/main/Kharatyan_Kocharov_2024_xcl_parsers.pdf)
at the
[Data-driven Approaches to Ancient Languages](https://www.vaia.be/en/courses/data-driven-approaches-to-ancient-languages)
workshop.



### English

For the biomed datasets, we use two sources of word vectors.

The `craft` and `genia` datasets used word vectors published by
bio.nlplab.org, specifically the `wikipedia-pubmed-and-PMC-w2v.bin` file
[available here](http://evexdb.org/pmresources/vec-space-models/)
trimmed to 100,000 vectors.

```
@inproceedings{Pyysalo2013DistributionalSR,
  title={Distributional Semantics Resources for Biomedical Text Processing},
  author={Sampo Pyysalo and Filip Ginter and Hans Moen and Tapio Salakoski and Sophia Ananiadou},
  year={2013},
  url={https://api.semanticscholar.org/CorpusID:3103489}
}
```

The `mimic` dataset used word vectors specifically trained on the Mimic data,
`BioWordVec_PubMed_MIMICIII_d200.vec.bin`, trimmed to 100,000 vectors,
[available here](https://github.com/ncbi-nlp/BioSentVec#biowordvec)

Zhang Y, Chen Q, Yang Z, Lin H, Lu Z. [BioWordVec, improving biomedical word embeddings with subword information and MeSH.](https://www.nature.com/articles/s41597-019-0055-0) Sci Data. 2019;6(1):52. Published 2019 May 10. doi:10.1038/s41597-019-0055-0

### Erzya

For Erzya, we used [vectors trained by Khalid Alnajjar](https://github.com/mokha/semantics) as part of a project to preserve endangered languages:

Alnajjar, K. (2021).
[When Word Embeddings Become Endangered](https://arxiv.org/abs/2103.13275).
In M. Hämäläinen, N. Partanen, & K. Alnajjar (Eds.),
Multilingual Facilitation (pp. 275-288).
University of Helsinki.
https://doi.org/10.31885/9789515150257

### Marathi

[The L3Cube project](https://github.com/l3cube-pune/MarathiNLP)
provides a couple very useful Marathi datasets we incorporate in our
models, NER and Sentiment.  They also provide a set of Fasttext WV.
We experimented with this to rebuild the POS, depparse, and NER
models.  In general, the results were not conclusive when compared
with the fasttextwiki model, so we kept fasttextwiki as the default.

In the following chart, POS and NER used the charlm as well, whereas
the depparse did not.

We compare the L3Cube MahaFT vectors with
[Fasttext for 157 languages](https://fasttext.cc/docs/en/crawl-vectors.html) and
[Fasttext from Wiki](https://fasttext.cc/docs/en/pretrained-vectors.html).
Fasttext 157 was best for depparse, Fasttext Wiki was best for NER, and L3Cube was best for POS.

| Pretrain     | UPOS  | UFeats | AllTags | UAS   | LAS   | CLAS  | MLAS  | BLEX  | NER dev | NER test |
| :------:     | :---: | :----: | :----:  | :--:  | :--:  | :--:  | :--:  | :--:  | :----:  | :---:    |
| fasttext     | 91.55 | 73.29  | 70.32   | 78.54 | 66.21 | 59.68 | 54.44 | 59.68 | 83.27   | 83.62    |
| fasttextwiki | 91.78 | 72.60  | 70.78   | 74.66 | 63.70 | 58.92 | 52.51 | 58.92 | 83.44   | 84.85    |
| l3cube       | 91.55 | 75.80  | 73.52   | 74.66 | 63.47 | 56.86 | 52.49 | 56.86 | 83.98   | 84.46    |

### Myanmar

Word vectors were kindly provided to us from UCSY in Myanmar.

Aye Mya Hlaing and Win Pa Pa (2020).
[Word Representations for Neural Network Based Myanmar Text-to-Speech System](http://oaji.net/articles/2020/3603-1582708892.pdf).
in International Journal of Intelligent Engineering and Systems, volume 13, pp 239-349.

### Old French

We use the vectors for French from the CoNLL 17 shared task.  Other suggestions are welcome.

### Sindhi

We trained a new set of word vectors using Glove.  For the training
corpus, we used Wikipedia, Oscar 2023, and a collection of text from
the NLP group at [ISRA University](https://isra.edu.pk) in Pakistan.
This is still a work in progress.