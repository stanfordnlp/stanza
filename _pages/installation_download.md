---
title: Installation & Model Download
keywords: installation-download
permalink: '/installation_download.html'
---

## Getting started

### Installation

To get started with StanfordNLP, we strongly recommend that you install it through [PyPI](https://pypi.org/). Once you have [pip installed](https://pip.pypa.io/en/stable/installing/), simply run in your command line

```bash
pip install stanfordnlp
```

This will take care of all of the dependencies necessary to run StanfordNLP. The neural pipeline of StanfordNLP depends on PyTorch 1.0.0 or a later version with compatible APIs.

**Note** StanfordNLP will not work with Python 3.5 or below. Please use Python 3.6 or above.

### Quick Example

To try out StanfordNLP, you can simply follow these steps in the interactive Python interpreter:

```python
>>> import stanfordnlp
>>> stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
>>> nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

The last command here will print out the words in the first sentence in the input string (or `Document`, as it is represented in StanfordNLP), as well as the indices for the word that governs it in the Universal Dependencies parse of that sentence (its "head"), along with the dependency relation between the words. The output should look like:

```
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')
```

To build a pipeline for other languages, simply pass in the language code to the constructor like this `stanfordnlp.Pipeline(lang="fr")`. For a full list of languages (and their corresponnding language codes) supported by StanfordNLP, please see [this section](#human-languages-supported-by-stanfordnlp).

We also provide a [demo script](https://github.com/stanfordnlp/stanfordnlp/blob/master/demo/pipeline_demo.py) in our Github repostory that demonstrates how one uses StanfordNLP in other languages than English, for example Chinese (traditional)

```python
python demo/pipeline_demo.py -l zh
```

And expect outputs like the following:

```
---
tokens of first sentence:
達沃斯	達沃斯	PROPN
世界	世界	NOUN
經濟	經濟	NOUN
論壇	論壇	NOUN
是	是	AUX
每年	每年	DET
全球	全球	NOUN
政	政	PART
商界	商界	NOUN
領袖	領袖	NOUN
聚	聚	VERB
在	在	VERB
一起	一起	NOUN
的	的	PART
年度	年度	NOUN
盛事	盛事	NOUN
。	。	PUNCT

---
dependency parse of first sentence:
('達沃斯', '4', 'nmod')
('世界', '4', 'nmod')
('經濟', '4', 'nmod')
('論壇', '16', 'nsubj')
('是', '16', 'cop')
('每年', '10', 'nmod')
('全球', '10', 'nmod')
('政', '9', 'case:pref')
('商界', '10', 'nmod')
('領袖', '11', 'nsubj')
('聚', '16', 'acl:relcl')
('在', '11', 'mark')
('一起', '11', 'obj')
('的', '11', 'mark:relcl')
('年度', '16', 'nmod')
('盛事', '0', 'root')
('。', '16', 'punct')
```

## Models for Human Languages

### Downloading and Using Models

Downloading models for human languages of your interest for use in the StanfordNLP pipeline is as simple as

```python
>>> import stanfordnlp
>>> stanfordnlp.download('ar')    # replace "ar" with the language code or treebank code you need, see below
```

The language code or treebank code can be looked up in the next section. If only the language code is specified, we will download the default models for that language (indicated by <i class="fas fa-check" style="color:#33a02c"></i> in the table), which are the models trained on the largest treebank available in that language. If you really care about the models of a specific treebank, you can also download the corresponding models with the treebank code.

To use the default model for any language, simply build the pipeline as follows

```python
>>> nlp = stanfordnlp.Pipeline(lang="es")    # replace "es" with the language of interest
```

If you are using a non-default treebank for the langauge, make sure to also specify the treebank code, for example

```python
>>> nlp = stanford.Pipeline(lang="it", treebank="it_postwita")
```

### Human Languages Supported by StanfordNLP

Below is a list of all of the (human) languages supported by StanfordNLP (through its neural pipeline). The performance of these systems on the [CoNLL 2018 Shared Task]() official test set (in our unofficial evaluation) can be found in the following subsection.

**Note**

1. Models marked with <i class="fas fa-exclamation-triangle" style="color:#e31a1c"></i> have significantly low unlabeled attachment score (UAS) when evaluated end-to-end (from tokenization all the way to dependency parsing). Specifically, their UAS is lower than 50% on the CoNLL 2018 Shared Task test set. Any use of these models for serious syntactical analysis is strongly discouraged.
2. <i class="fas fa-flag" style="color:#fdae61"></i> marks models that are at least 1% absolute UAS worse than the full neural pipeline presented in [our paper](https://nlp.stanford.edu/pubs/qi2018universal.pdf) (which uses the [Tensorflow counterparts](https://github.com/tdozat/Parser-v3) for the tagger and the parser), so that might raise a yellow flag.

| Language | Treebank | Language code | Treebank code | Models | Version | Notes |
| :------- | :------- | :------------ | :------------ | :----- | :------ | :---- |
| Afrikaans | AfriBooms | af | af_afribooms | [download](http://nlp.stanford.edu/software/conll_2018/af_afribooms_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-flag" style="color:#fdae61"></i> |
| Ancient Greek | Perseus | grc | grc_perseus | [download](http://nlp.stanford.edu/software/conll_2018/grc_perseus_models.zip) | 0.1.0 |  |
|  | PROIEL | grc | grc_proiel | [download](http://nlp.stanford.edu/software/conll_2018/grc_proiel_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Arabic | PADT | ar | ar_padt | [download](http://nlp.stanford.edu/software/conll_2018/ar_padt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Armenian | ArmTDP | hy | hy_armtdp | [download](http://nlp.stanford.edu/software/conll_2018/hy_armtdp_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-flag" style="color:#fdae61"></i> |
| Basque | BDT | eu | eu_bdt | [download](http://nlp.stanford.edu/software/conll_2018/eu_bdt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Bulgarian | BTB | bg | bg_btb | [download](http://nlp.stanford.edu/software/conll_2018/bg_btb_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Buryat | BDT | bxr | bxr_bdt | [download](http://nlp.stanford.edu/software/conll_2018/bxr_bdt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-exclamation-triangle" style="color:#e31a1c"></i> |
| Catalan | AnCora | ca | ca_ancora | [download](http://nlp.stanford.edu/software/conll_2018/ca_ancora_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Chinese (traditional) | GSD | zh | zh_gsd | [download](http://nlp.stanford.edu/software/conll_2018/zh_gsd_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Croatian | SET | hr | hr_set | [download](http://nlp.stanford.edu/software/conll_2018/hr_set_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Czech | CAC | cs | cs_cac | [download](http://nlp.stanford.edu/software/conll_2018/cs_cac_models.zip) | 0.1.0 |  |
|  | FicTree | cs | cs_fictree | [download](http://nlp.stanford.edu/software/conll_2018/cs_fictree_models.zip) | 0.1.0 |  |
|  | PDT | cs | cs_pdt | [download](http://nlp.stanford.edu/software/conll_2018/cs_pdt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Danish | DDT | da | da_ddt | [download](http://nlp.stanford.edu/software/conll_2018/da_ddt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-flag" style="color:#fdae61"></i> |
| Dutch | Alpino | nl | nl_alpino | [download](http://nlp.stanford.edu/software/conll_2018/nl_alpino_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | LassySmall | nl | nl_lassysmall | [download](http://nlp.stanford.edu/software/conll_2018/nl_lassysmall_models.zip) | 0.1.0 |  |
| English | EWT | en | en_ewt | [download](http://nlp.stanford.edu/software/conll_2018/en_ewt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | GUM | en | en_gum | [download](http://nlp.stanford.edu/software/conll_2018/en_gum_models.zip) | 0.1.0 |  |
|  | LinES | en | en_lines | [download](http://nlp.stanford.edu/software/conll_2018/en_lines_models.zip) | 0.1.0 |  |
| Estonian | EDT | et | et_edt | [download](http://nlp.stanford.edu/software/conll_2018/et_edt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Finnish | FTB | fi | fi_ftb | [download](http://nlp.stanford.edu/software/conll_2018/fi_ftb_models.zip) | 0.1.0 |  |
|  | TDT | fi | fi_tdt | [download](http://nlp.stanford.edu/software/conll_2018/fi_tdt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-flag" style="color:#fdae61"></i> |
| French | GSD | fr | fr_gsd | [download](http://nlp.stanford.edu/software/conll_2018/fr_gsd_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | Sequoia | fr | fr_sequoia | [download](http://nlp.stanford.edu/software/conll_2018/fr_sequoia_models.zip) | 0.1.0 |  |
|  | Spoken | fr | fr_spoken | [download](http://nlp.stanford.edu/software/conll_2018/fr_spoken_models.zip) | 0.1.0 |  <i class="fas fa-flag" style="color:#fdae61"></i> |
| Galician | CTG | gl | gl_ctg | [download](http://nlp.stanford.edu/software/conll_2018/gl_ctg_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | TreeGal | gl | gl_treegal | [download](http://nlp.stanford.edu/software/conll_2018/gl_treegal_models.zip) | 0.1.0 |  <i class="fas fa-flag" style="color:#fdae61"></i> |
| German | GSD | de | de_gsd | [download](http://nlp.stanford.edu/software/conll_2018/de_gsd_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Gothic | PROIEL | got | got_proiel | [download](http://nlp.stanford.edu/software/conll_2018/got_proiel_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Greek | GDT | el | el_gdt | [download](http://nlp.stanford.edu/software/conll_2018/el_gdt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Hebrew | HTB | he | he_htb | [download](http://nlp.stanford.edu/software/conll_2018/he_htb_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Hindi | HDTB | hi | hi_hdtb | [download](http://nlp.stanford.edu/software/conll_2018/hi_hdtb_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Hungarian | Szeged | hu | hu_szeged | [download](http://nlp.stanford.edu/software/conll_2018/hu_szeged_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Indonesian | GSD | id | id_gsd | [download](http://nlp.stanford.edu/software/conll_2018/id_gsd_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Irish | IDT | ga | ga_idt | [download](http://nlp.stanford.edu/software/conll_2018/ga_idt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Italian | ISDT | it | it_isdt | [download](http://nlp.stanford.edu/software/conll_2018/it_isdt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-flag" style="color:#fdae61"></i> |
|  | PoSTWITA | it | it_postwita | [download](http://nlp.stanford.edu/software/conll_2018/it_postwita_models.zip) | 0.1.0 |  |
| Japanese | GSD | ja | ja_gsd | [download](http://nlp.stanford.edu/software/conll_2018/ja_gsd_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-flag" style="color:#fdae61"></i> |
| Kazakh | KTB | kk | kk_ktb | [download](http://nlp.stanford.edu/software/conll_2018/kk_ktb_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-exclamation-triangle" style="color:#e31a1c"></i> |
| Korean | GSD | ko | ko_gsd | [download](http://nlp.stanford.edu/software/conll_2018/ko_gsd_models.zip) | 0.1.0 |  |
|  | Kaist | ko | ko_kaist | [download](http://nlp.stanford.edu/software/conll_2018/ko_kaist_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Kurmanji | MG | kmr | kmr_mg | [download](http://nlp.stanford.edu/software/conll_2018/kmr_mg_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-exclamation-triangle" style="color:#e31a1c"></i> <i class="fas fa-flag" style="color:#fdae61"></i> |
| Latin | ITTB | la | la_ittb | [download](http://nlp.stanford.edu/software/conll_2018/la_ittb_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | Perseus | la | la_perseus | [download](http://nlp.stanford.edu/software/conll_2018/la_perseus_models.zip) | 0.1.0 |  |
|  | PROIEL | la | la_proiel | [download](http://nlp.stanford.edu/software/conll_2018/la_proiel_models.zip) | 0.1.0 |  |
| Latvian | LVTB | lv | lv_lvtb | [download](http://nlp.stanford.edu/software/conll_2018/lv_lvtb_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| North Sami | Giella | sme | sme_giella | [download](http://nlp.stanford.edu/software/conll_2018/sme_giella_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Norwegian | Bokmaal | no_bokmaal | no_bokmaal | [download](http://nlp.stanford.edu/software/conll_2018/no_bokmaal_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | Nynorsk | no_nynorsk | no_nynorsk | [download](http://nlp.stanford.edu/software/conll_2018/no_nynorsk_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | NynorskLIA | no_nynorsk | no_nynorsklia | [download](http://nlp.stanford.edu/software/conll_2018/no_nynorsklia_models.zip) | 0.1.0 |  |
| Old Church Slavonic | PROIEL | cu | cu_proiel | [download](http://nlp.stanford.edu/software/conll_2018/cu_proiel_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Old French | SRCMF | fro | fro_srcmf | [download](http://nlp.stanford.edu/software/conll_2018/fro_srcmf_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Persian | Seraji | fa | fa_seraji | [download](http://nlp.stanford.edu/software/conll_2018/fa_seraji_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Polish | LFG | pl | pl_lfg | [download](http://nlp.stanford.edu/software/conll_2018/pl_lfg_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | SZ | pl | pl_sz | [download](http://nlp.stanford.edu/software/conll_2018/pl_sz_models.zip) | 0.1.0 |  |
| Portuguese | Bosque | pt | pt_bosque | [download](http://nlp.stanford.edu/software/conll_2018/pt_bosque_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Romanian | RRT | ro | ro_rrt | [download](http://nlp.stanford.edu/software/conll_2018/ro_rrt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Russian | SynTagRus | ru | ru_syntagrus | [download](http://nlp.stanford.edu/software/conll_2018/ru_syntagrus_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | Taiga | ru | ru_taiga | [download](http://nlp.stanford.edu/software/conll_2018/ru_taiga_models.zip) | 0.1.0 |  <i class="fas fa-flag" style="color:#fdae61"></i> |
| Serbian | SET | sr | sr_set | [download](http://nlp.stanford.edu/software/conll_2018/sr_set_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Slovak | SNK | sk | sk_snk | [download](http://nlp.stanford.edu/software/conll_2018/sk_snk_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Slovenian | SSJ | sl | sl_ssj | [download](http://nlp.stanford.edu/software/conll_2018/sl_ssj_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
|  | SST | sl | sl_sst | [download](http://nlp.stanford.edu/software/conll_2018/sl_sst_models.zip) | 0.1.0 |  |
| Spanish | AnCora | es | es_ancora | [download](http://nlp.stanford.edu/software/conll_2018/es_ancora_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Swedish | LinES | sv | sv_lines | [download](http://nlp.stanford.edu/software/conll_2018/sv_lines_models.zip) | 0.1.0 |  |
|  | Talbanken | sv | sv_talbanken | [download](http://nlp.stanford.edu/software/conll_2018/sv_talbanken_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Turkish | IMST | tr | tr_imst | [download](http://nlp.stanford.edu/software/conll_2018/tr_imst_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Ukrainian | IU | uk | uk_iu | [download](http://nlp.stanford.edu/software/conll_2018/uk_iu_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Upper Sorbian | UFAL | hsb | hsb_ufal | [download](http://nlp.stanford.edu/software/conll_2018/hsb_ufal_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> <i class="fas fa-exclamation-triangle" style="color:#e31a1c"></i> <i class="fas fa-flag" style="color:#fdae61"></i> |
| Urdu | UDTB | ur | ur_udtb | [download](http://nlp.stanford.edu/software/conll_2018/ur_udtb_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Uyghur | UDT | ug | ug_udt | [download](http://nlp.stanford.edu/software/conll_2018/ug_udt_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |
| Vietnamese | VTB | vi | vi_vtb | [download](http://nlp.stanford.edu/software/conll_2018/vi_vtb_models.zip) | 0.1.0 |  <i class="fas fa-check" style="color:#33a02c"></i> |

### System Performance in CoNLL 2018 Shared Task

In the table below you can find the performance of the version 0.1.0 models of StanfordNLP's neural pipeline, which is our best attempt at replicating our final system in the full PyTorch pipeline. The scores shown are from an end-to-end evaluation on the official test sets (from raw text to the full CoNLL-U file), and the scores are generated with the official evaluation script. For how we handled treebanks with no training data, please refer to the [system description paper](https://nlp.stanford.edu/pubs/qi2018universal.pdf) for details. 

These results are "unofficial" in the sense that these don't reflect the systems' state at the time of submission to the Shared Task (which had an unfortunate bug that we thereafter have fixed, as is also described in [our paper](https://nlp.stanford.edu/pubs/qi2018universal.pdf)).

|  | Tokens | Sentences | Words | UPOS | XPOS | UFeats | AllTags | Lemmas | UAS | LAS | CLAS | MLAS | BLEX |
| :-- | :----- | :-------- | :---- | :--- | :--- | :----- | :------ | :----- | :-- | :-- | :--- | :--- | :--- |
| UD\_<wbr>Afrikaans-AfriBooms | 99.67 | 100 | 99.67 | 97.47 | 93.89 | 97.01 | 93.85 | 96.78 | 87.68 | 84.69 | 79.07 | 75.02 | 75.06 |
| UD\_<wbr>Ancient\_<wbr>Greek-Perseus | 99.97 | 98.81 | 99.97 | 92.44 | 85.06 | 91.03 | 84.82 | 87.78 | 78.81 | 73.44 | 67.92 | 53.58 | 56.71 |
| UD\_<wbr>Ancient\_<wbr>Greek-PROIEL | 100 | 51.98 | 100 | 97.21 | 97.69 | 92.35 | 91.03 | 96.42 | 80.89 | 77.04 | 72.29 | 61.95 | 69.63 |
| UD\_<wbr>Arabic-PADT | 99.98 | 80.4 | 96.53 | 93.47 | 90.78 | 90.89 | 90.5 | 91.54 | 81.29 | 76.99 | 73.97 | 68.7 | 70.13 |
| UD\_<wbr>Armenian-ArmTDP | 95.66 | 87.45 | 95.09 | 72.82 | 95.09 | 55.67 | 53.47 | 67.84 | 50.58 | 31.47 | 23.82 | 11.31 | 14.85 |
| UD\_<wbr>Basque-BDT | 99.99 | 99.92 | 99.99 | 96.09 | 99.99 | 93.14 | 91.33 | 95.28 | 85.86 | 82.59 | 81.06 | 73.1 | 76.57 |
| UD\_<wbr>Breton-KEB | 91.66 | 89.54 | 90.9 | 33.37 | 0 | 21.34 | 0 | 50.1 | 34.85 | 11.25 | 7.07 | 0.11 | 1.77 |
| UD\_<wbr>Bulgarian-BTB | 99.93 | 97.12 | 99.93 | 98.72 | 96.56 | 97.56 | 95.93 | 96.56 | 93.31 | 90.18 | 86.92 | 83.57 | 82.72 |
| UD\_<wbr>Buryat-BDT | 96.45 | 90.62 | 96.45 | 41.96 | 96.45 | 33 | 24.99 | 53.19 | 28.96 | 12.47 | 7.77 | 0.94 | 3.05 |
| UD\_<wbr>Catalan-AnCora | 99.99 | 99.84 | 99.98 | 98.75 | 98.75 | 98.13 | 97.62 | 97.73 | 92.69 | 90.44 | 86.24 | 83.32 | 84.05 |
| UD\_<wbr>Chinese-GSD | 92.6 | 99.1 | 92.6 | 88.87 | 88.71 | 91.89 | 87.93 | 92.6 | 74.25 | 70.91 | 67.18 | 62.7 | 67.18 |
| UD\_<wbr>Croatian-SET | 99.91 | 97.59 | 99.91 | 97.94 | 99.91 | 91.98 | 91.43 | 95.41 | 91.13 | 86.58 | 83.8 | 73.67 | 78.92 |
| UD\_<wbr>Czech-CAC | 99.99 | 100 | 99.97 | 98.89 | 95.11 | 93.75 | 92.95 | 97.46 | 91.92 | 89.41 | 87.02 | 80.24 | 84.21 |
| UD\_<wbr>Czech-FicTree | 99.98 | 99.19 | 99.97 | 98.31 | 95.51 | 96.31 | 94.91 | 97.9 | 93.68 | 90.91 | 88.46 | 82.83 | 85.76 |
| UD\_<wbr>Czech-PDT | 99.98 | 94.06 | 99.97 | 98.59 | 95.51 | 94.83 | 93.92 | 98.22 | 91.85 | 89.63 | 87.87 | 81.76 | 85.92 |
| UD\_<wbr>Czech-PUD | 99.15 | 95.5 | 99 | 96.13 | 92.29 | 91.66 | 90.06 | 95.45 | 89.49 | 84.45 | 81.87 | 73.54 | 79.18 |
| UD\_<wbr>Danish-DDT | 99.96 | 93.06 | 99.96 | 97.74 | 99.96 | 97.5 | 96.53 | 95.62 | 86.31 | 83.71 | 80.64 | 76.33 | 75.58 |
| UD\_<wbr>Dutch-Alpino | 99.96 | 90.86 | 99.96 | 96.05 | 94.18 | 95.88 | 93.55 | 96.27 | 90.02 | 87 | 81.82 | 75.39 | 76.81 |
| UD\_<wbr>Dutch-LassySmall | 99.92 | 78.85 | 99.92 | 96.34 | 95 | 96.27 | 94.38 | 96.33 | 86.89 | 83.39 | 77.32 | 73.4 | 73.84 |
| UD\_<wbr>English-EWT | 99.16 | 81.27 | 99.16 | 95.66 | 95.36 | 96.21 | 94.19 | 96.5 | 86.46 | 83.8 | 80.35 | 76.08 | 77.67 |
| UD\_<wbr>English-GUM | 99.81 | 81.76 | 99.81 | 95.67 | 95.6 | 96.66 | 94.74 | 96.17 | 85.37 | 82.03 | 76.53 | 72.29 | 72.26 |
| UD\_<wbr>English-LinES | 99.92 | 88.03 | 99.92 | 96.84 | 95.69 | 96.93 | 93.62 | 96.3 | 83.08 | 78.37 | 75.52 | 71.12 | 71.48 |
| UD\_<wbr>English-PUD | 99.86 | 96.15 | 99.86 | 95.91 | 94.8 | 95.61 | 92.06 | 95.25 | 88.8 | 86.13 | 83.47 | 75.62 | 78.17 |
| UD\_<wbr>Estonian-EDT | 99.95 | 93.82 | 99.95 | 97.12 | 97.97 | 95.75 | 94.4 | 95 | 86.89 | 83.95 | 82.31 | 77.17 | 77.08 |
| UD\_<wbr>Faroese-OFT | 99.54 | 84.4 | 99.54 | 59.82 | 0 | 31.35 | 0 | 37.45 | 51.14 | 41.54 | 34.32 | 0.26 | 13.37 |
| UD\_<wbr>Finnish-FTB | 100 | 90.24 | 99.97 | 96.03 | 95.32 | 96.54 | 94.33 | 95.27 | 89.84 | 87.23 | 84.46 | 80.16 | 80.28 |
| UD\_<wbr>Finnish-PUD | 99.46 | 91.25 | 99.46 | 97.3 | 0 | 96.5 | 0 | 92.56 | 90.02 | 88.02 | 86.34 | 82.23 | 78.6 |
| UD\_<wbr>Finnish-TDT | 99.76 | 91.82 | 99.76 | 97.03 | 97.78 | 95.47 | 94.6 | 93.19 | 88.72 | 86.33 | 84.81 | 79.8 | 77.68 |
| UD\_<wbr>French-GSD | 99.67 | 95.4 | 99.37 | 96.9 | 99.37 | 96.49 | 95.66 | 96.94 | 89.3 | 86.53 | 82.64 | 78.11 | 79.59 |
| UD\_<wbr>French-Sequoia | 99.85 | 90.61 | 99.65 | 98.45 | 99.65 | 97.83 | 97.37 | 97.44 | 90.8 | 88.91 | 85.67 | 82.82 | 82.73 |
| UD\_<wbr>French-Spoken | 100 | 19.59 | 100 | 96.36 | 97.45 | 100 | 93.95 | 94.68 | 75.82 | 70.18 | 61.78 | 59.16 | 58.31 |
| UD\_<wbr>Galician-CTG | 99.89 | 98.26 | 99.37 | 97.27 | 97.07 | 99.21 | 96.8 | 97.46 | 85.2 | 82.6 | 77.03 | 71.04 | 74.84 |
| UD\_<wbr>Galician-TreeGal | 99.46 | 88.5 | 98.22 | 93.9 | 91.64 | 92.99 | 90.59 | 93.02 | 76.61 | 71.36 | 63.81 | 57.32 | 58.06 |
| UD\_<wbr>German-GSD | 99.55 | 85.07 | 99.55 | 93.89 | 96.88 | 89.59 | 84.3 | 95.97 | 83.62 | 79.17 | 75.22 | 58.03 | 70.68 |
| UD\_<wbr>Gothic-PROIEL | 100 | 40.54 | 100 | 96.19 | 96.74 | 90.24 | 88.57 | 95.77 | 76.87 | 70.74 | 68.21 | 58.69 | 65.64 |
| UD\_<wbr>Greek-GDT | 99.91 | 92.74 | 99.92 | 97.84 | 97.84 | 94.99 | 94.3 | 95.59 | 91.33 | 88.97 | 84.53 | 78.43 | 78.52 |
| UD\_<wbr>Hebrew-HTB | 99.98 | 100 | 93.32 | 90.63 | 90.63 | 89.21 | 88.57 | 89.99 | 79.37 | 75.76 | 69.11 | 63.27 | 65.19 |
| UD\_<wbr>Hindi-HDTB | 100 | 99.35 | 100 | 97.45 | 96.98 | 93.85 | 91.9 | 96.42 | 94.73 | 91.65 | 87.94 | 78.07 | 86.37 |
| UD\_<wbr>Hungarian-Szeged | 99.8 | 96.54 | 99.8 | 96.05 | 99.8 | 93.94 | 93.13 | 91.67 | 83.41 | 78.58 | 77.12 | 69.12 | 68.74 |
| UD\_<wbr>Indonesian-GSD | 100 | 93.94 | 100 | 93.67 | 94.62 | 95.78 | 88.99 | 99.63 | 85.7 | 79.38 | 77.43 | 68.93 | 77.09 |
| UD\_<wbr>Irish-IDT | 99.71 | 96.69 | 99.71 | 92.49 | 91.19 | 82.63 | 79.2 | 89.15 | 79.28 | 70.27 | 60.89 | 45.14 | 52.22 |
| UD\_<wbr>Italian-ISDT | 99.9 | 98.76 | 99.77 | 97.99 | 97.91 | 97.82 | 97.17 | 97.35 | 92.33 | 90.51 | 86.08 | 83.25 | 82.62 |
| UD\_<wbr>Italian-PoSTWITA | 99.73 | 62.97 | 99.41 | 96.18 | 96.04 | 96.18 | 94.93 | 95.45 | 82.87 | 78.49 | 72.39 | 68.77 | 68.79 |
| UD\_<wbr>Japanese-GSD | 92.24 | 94.92 | 92.24 | 90.76 | 92.24 | 92.23 | 90.76 | 91.53 | 79.98 | 78.12 | 67.92 | 66.47 | 67.35 |
| UD\_<wbr>Japanese-Modern | 70.39 | 0 | 70.39 | 53 | 0 | 68.45 | 0 | 57.82 | 34.82 | 28.26 | 15.83 | 12.15 | 14.07 |
| UD\_<wbr>Kazakh-KTB | 95.13 | 81.4 | 95.67 | 57.9 | 58.22 | 43.68 | 37.17 | 55.29 | 45.05 | 26.25 | 20.62 | 8.22 | 9.58 |
| UD\_<wbr>Korean-GSD | 99.9 | 96.59 | 99.9 | 96.17 | 90.97 | 99.67 | 88.69 | 92.24 | 88.02 | 84.16 | 82.04 | 79.65 | 75.43 |
| UD\_<wbr>Korean-Kaist | 100 | 99.93 | 100 | 95.6 | 86.64 | 100 | 86.64 | 92.47 | 88.48 | 86.58 | 84.21 | 80.89 | 77.19 |
| UD\_<wbr>Kurmanji-MG | 94.12 | 88.61 | 93.79 | 55.1 | 54.3 | 41.12 | 35.87 | 56.24 | 33.39 | 24.18 | 18.13 | 4.3 | 8.49 |
| UD\_<wbr>Latin-ITTB | 99.69 | 79.95 | 99.69 | 98.08 | 94.61 | 96.25 | 93.53 | 98.31 | 87.71 | 85.18 | 83.77 | 79.7 | 83.02 |
| UD\_<wbr>Latin-Perseus | 100 | 99.31 | 100 | 90.51 | 77.83 | 82.1 | 77.4 | 79.95 | 72.48 | 62.04 | 57.45 | 45.02 | 43.87 |
| UD\_<wbr>Latin-PROIEL | 100 | 42.57 | 100 | 96.91 | 97.12 | 90.89 | 90.09 | 95.85 | 76.22 | 71.93 | 69.48 | 60.22 | 66.92 |
| UD\_<wbr>Latvian-LVTB | 99.71 | 98.38 | 99.71 | 94.99 | 87.11 | 91.92 | 86.34 | 92.17 | 86.06 | 81.78 | 78.86 | 69.05 | 71.28 |
| UD\_<wbr>Naija-NSC | 95.27 | 2.03 | 95.27 | 54.49 | 0 | 36.35 | 0 | 89.22 | 36.25 | 22.59 | 24.02 | 3.51 | 22.35 |
| UD\_<wbr>North\_<wbr>Sami-Giella | 99.76 | 99.65 | 99.76 | 91.72 | 93.16 | 88.5 | 84.85 | 88.08 | 75.61 | 69.66 | 66.82 | 60.09 | 57.72 |
| UD\_<wbr>Norwegian-Bokmaal | 99.85 | 97.78 | 99.85 | 98.14 | 99.85 | 97.12 | 96.33 | 97.48 | 91.87 | 90.1 | 87.82 | 83.82 | 84.63 |
| UD\_<wbr>Norwegian-Nynorsk | 99.95 | 95 | 99.95 | 97.84 | 99.95 | 96.69 | 95.85 | 97.03 | 91.61 | 89.58 | 87.34 | 82.38 | 83.58 |
| UD\_<wbr>Norwegian-NynorskLIA | 100 | 100 | 100 | 91.39 | 100 | 90.54 | 86.43 | 93.3 | 65.49 | 57.3 | 51.69 | 45.14 | 47.12 |
| UD\_<wbr>Old\_<wbr>Church\_<wbr>Slavonic-PROIEL | 100 | 48.48 | 100 | 96.34 | 96.64 | 90.78 | 89.78 | 95.01 | 80.6 | 75.73 | 74.53 | 65.9 | 71.4 |
| UD\_<wbr>Old\_<wbr>French-SRCMF | 100 | 100 | 100 | 96.01 | 96.04 | 97.81 | 95.54 | 100 | 91.77 | 86.86 | 83.84 | 80.6 | 83.84 |
| UD\_<wbr>Persian-Seraji | 100 | 99.25 | 99.66 | 97.32 | 97.24 | 97.37 | 96.89 | 97.46 | 89.84 | 86.55 | 83.3 | 81.34 | 81.27 |
| UD\_<wbr>Polish-LFG | 99.94 | 99.91 | 99.94 | 98.67 | 94.69 | 95.63 | 94.08 | 95.93 | 96.56 | 94.71 | 93.02 | 87.73 | 87.86 |
| UD\_<wbr>Polish-SZ | 100 | 99.46 | 99.9 | 97.87 | 92.74 | 92.67 | 91.52 | 94.65 | 93.05 | 90.8 | 89.13 | 80.99 | 82.55 |
| UD\_<wbr>Portuguese-Bosque | 99.71 | 93.14 | 99.55 | 96.69 | 99.55 | 96.31 | 94.24 | 96.9 | 89.85 | 87.62 | 83.51 | 76.47 | 79.91 |
| UD\_<wbr>Romanian-RRT | 99.79 | 95.28 | 99.79 | 97.44 | 96.86 | 97.06 | 96.66 | 97.02 | 90.83 | 86.24 | 82.3 | 78.44 | 79.09 |
| UD\_<wbr>Russian-SynTagRus | 99.61 | 99.1 | 99.61 | 98.32 | 99.61 | 96.23 | 95.91 | 97.52 | 92.84 | 91.2 | 89.56 | 85.53 | 87.19 |
| UD\_<wbr>Russian-Taiga | 97.33 | 77.58 | 97.33 | 91.92 | 97.3 | 84.94 | 83.42 | 84.36 | 70.01 | 63.16 | 59.33 | 48.61 | 48.77 |
| UD\_<wbr>Serbian-SET | 99.96 | 95.32 | 99.96 | 98.16 | 99.96 | 93.81 | 93.55 | 95.01 | 92.3 | 88.83 | 86.38 | 78.37 | 80.45 |
| UD\_<wbr>Slovak-SNK | 99.98 | 90.34 | 99.98 | 96.51 | 86.89 | 91.3 | 86.19 | 93.35 | 90.41 | 87.33 | 85.23 | 75.22 | 77.92 |
| UD\_<wbr>Slovenian-SSJ | 99.9 | 98.86 | 99.9 | 98.42 | 95.11 | 95.35 | 94.62 | 95.78 | 92.89 | 91.07 | 88.39 | 83.09 | 82.91 |
| UD\_<wbr>Slovenian-SST | 100 | 26.8 | 100 | 93.97 | 88.02 | 88.27 | 85.83 | 92.09 | 62.98 | 56.24 | 51.53 | 45.13 | 47.13 |
| UD\_<wbr>Spanish-AnCora | 99.96 | 99.39 | 99.95 | 98.59 | 98.59 | 98.13 | 97.51 | 98.08 | 91.96 | 89.85 | 85.85 | 82.99 | 83.65 |
| UD\_<wbr>Swedish-LinES | 99.93 | 85.85 | 99.93 | 96.71 | 94.7 | 89.48 | 86.69 | 95.37 | 85.25 | 80.84 | 79.18 | 65.87 | 73.89 |
| UD\_<wbr>Swedish-PUD | 98.94 | 89.38 | 98.94 | 93.8 | 91.97 | 78.33 | 76.38 | 84.3 | 82.46 | 78.35 | 76 | 51.08 | 61.33 |
| UD\_<wbr>Swedish-Talbanken | 99.97 | 98.11 | 99.97 | 97.63 | 96.5 | 96.71 | 95.53 | 97.01 | 89.04 | 85.84 | 83.46 | 78.97 | 79.94 |
| UD\_<wbr>Thai-PUD | 8.56 | 0.39 | 8.56 | 5.86 | 0.02 | 5.67 | 0 | 8.56 | 0.84 | 0.7 | 0.42 | 0.03 | 0.42 |
| UD\_<wbr>Turkish-IMST | 99.85 | 98.11 | 97.87 | 94.25 | 93.47 | 92.01 | 90.42 | 93.88 | 70.94 | 64.76 | 62.11 | 56.62 | 59.6 |
| UD\_<wbr>Ukrainian-IU | 99.7 | 97.58 | 99.7 | 96.73 | 92.12 | 91.91 | 90.8 | 94.31 | 87.34 | 84.13 | 80.47 | 73.28 | 74.55 |
| UD\_<wbr>Upper\_<wbr>Sorbian-UFAL | 92.7 | 84.17 | 92.7 | 55.99 | 92.7 | 37.28 | 35.38 | 55.71 | 33.14 | 23.61 | 16.71 | 3.13 | 7.49 |
| UD\_<wbr>Urdu-UDTB | 99.98 | 98.88 | 99.98 | 94.54 | 92.73 | 84.23 | 80.55 | 95.47 | 88.34 | 82.54 | 76.43 | 59.07 | 74.07 |
| UD\_<wbr>Uyghur-UDT | 99.82 | 85.4 | 99.82 | 89.34 | 91.7 | 87.81 | 80.16 | 95.68 | 75.53 | 63.55 | 56.95 | 45.49 | 53.86 |
| UD\_<wbr>Vietnamese-VTB | 87.18 | 93.15 | 87.18 | 79.48 | 77.82 | 86.97 | 77.79 | 87.12 | 53.05 | 47.56 | 44.35 | 41.37 | 44.3 |