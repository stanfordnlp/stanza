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
