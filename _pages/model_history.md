---
layout: page
title: Model History
keywords: models history
permalink: '/model_history.html'
nav_order: 4
parent: Resources
---

This page contains links to downloadable models for all historical versions of Stanza. Note that model versions prior to 0.2.0 (inclusive) require the StanfordNLP package (`pip install stanfordnlp`), and those after require the Stanza package (`pip install stanza`).

### 1.5.1

All UD models rebuilt with UD 2.12

### 1.4.0

Upgraded English constituency parser and other conparse models.  NER models for JA, MY

### 1.3.0

English constituency parser added.  No charlm or bert integration yet,
so the scores were quite low relative to state of the art.
Multilingual langid model added.

All UD models rebuilt with UD 2.8

### 1.2.3

NER models for AF, IT.

### 1.2.1

NER models for BG, HU, FI, VI, and all tokenize/mwt/lemma/pos/depparse models updated to UD 2.8.

### 1.1.1

This version extends Stanza's v1.0.0 models with additional [sentiment analysis models](sentiment) for English, German and Chinese pipelines, a new [tokenizer for the Thai language](available_models#other-available-models-for-tokenization), and new suite of [Biomedical and Clinical English syntactic analysis and NER model packages](available_biomed_models). We've changed the default packages for a few languages for robustness. These changes include: Polish (default is now `PDB` model, from previous `LFG`), Korean (default is now `GSD`, from previous Kaist), Lithuanian (default is now `ALKSNIS`, from previous HSE).

### 1.0.0

For models used by Stanza v1.0.0 (latest version), you can find all related information on the [Usage: Available Models](models) page.

### 0.2.0

| Language | Treebank | Language code | Treebank code | Models |
| :------- | :------- | :------------ | :------------ | :----- |
| Afrikaans | AfriBooms | af | af_afribooms | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/af_afribooms_models.zip) |
| Ancient Greek | Perseus | grc | grc_perseus | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/grc_perseus_models.zip) |
|  | PROIEL | grc | grc_proiel | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/grc_proiel_models.zip) |
| Arabic | PADT | ar | ar_padt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ar_padt_models.zip) |
| Armenian | ArmTDP | hy | hy_armtdp | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/hy_armtdp_models.zip) |
| Basque | BDT | eu | eu_bdt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/eu_bdt_models.zip) |
| Bulgarian | BTB | bg | bg_btb | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/bg_btb_models.zip) |
| Buryat | BDT | bxr | bxr_bdt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/bxr_bdt_models.zip) |
| Catalan | AnCora | ca | ca_ancora | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ca_ancora_models.zip) |
| Chinese (traditional) | GSD | zh | zh_gsd | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/zh_gsd_models.zip) |
| Croatian | SET | hr | hr_set | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/hr_set_models.zip) |
| Czech | CAC | cs | cs_cac | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/cs_cac_models.zip) |
|  | FicTree | cs | cs_fictree | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/cs_fictree_models.zip) |
|  | PDT | cs | cs_pdt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/cs_pdt_models.zip) |
| Danish | DDT | da | da_ddt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/da_ddt_models.zip) |
| Dutch | Alpino | nl | nl_alpino | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/nl_alpino_models.zip) |
|  | LassySmall | nl | nl_lassysmall | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/nl_lassysmall_models.zip) |
| English | EWT | en | en_ewt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/en_ewt_models.zip) |
|  | GUM | en | en_gum | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/en_gum_models.zip) |
|  | LinES | en | en_lines | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/en_lines_models.zip) |
| Estonian | EDT | et | et_edt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/et_edt_models.zip) |
| Finnish | FTB | fi | fi_ftb | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/fi_ftb_models.zip) |
|  | TDT | fi | fi_tdt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/fi_tdt_models.zip) |
| French | GSD | fr | fr_gsd | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/fr_gsd_models.zip) |
|  | Sequoia | fr | fr_sequoia | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/fr_sequoia_models.zip) |
|  | Spoken | fr | fr_spoken | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/fr_spoken_models.zip) |
| Galician | CTG | gl | gl_ctg | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/gl_ctg_models.zip) |
|  | TreeGal | gl | gl_treegal | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/gl_treegal_models.zip) |
| German | GSD | de | de_gsd | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/de_gsd_models.zip) |
| Gothic | PROIEL | got | got_proiel | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/got_proiel_models.zip) |
| Greek | GDT | el | el_gdt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/el_gdt_models.zip) |
| Hebrew | HTB | he | he_htb | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/he_htb_models.zip) |
| Hindi | HDTB | hi | hi_hdtb | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/hi_hdtb_models.zip) |
| Hungarian | Szeged | hu | hu_szeged | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/hu_szeged_models.zip) |
| Indonesian | GSD | id | id_gsd | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/id_gsd_models.zip) |
| Irish | IDT | ga | ga_idt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ga_idt_models.zip) |
| Italian | ISDT | it | it_isdt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/it_isdt_models.zip) |
|  | PoSTWITA | it | it_postwita | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/it_postwita_models.zip) |
| Japanese | GSD | ja | ja_gsd | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ja_gsd_models.zip) |
| Kazakh | KTB | kk | kk_ktb | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/kk_ktb_models.zip) |
| Korean | GSD | ko | ko_gsd | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ko_gsd_models.zip) |
|  | Kaist | ko | ko_kaist | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ko_kaist_models.zip) |
| Kurmanji | MG | kmr | kmr_mg | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/kmr_mg_models.zip) |
| Latin | ITTB | la | la_ittb | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/la_ittb_models.zip) |
|  | Perseus | la | la_perseus | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/la_perseus_models.zip) |
|  | PROIEL | la | la_proiel | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/la_proiel_models.zip) |
| Latvian | LVTB | lv | lv_lvtb | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/lv_lvtb_models.zip) |
| North Sami | Giella | sme | sme_giella | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/sme_giella_models.zip) |
| Norwegian | Bokmaal | no_bokmaal | no_bokmaal | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/no_bokmaal_models.zip) |
|  | Nynorsk | no_nynorsk | no_nynorsk | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/no_nynorsk_models.zip) |
|  | NynorskLIA | no_nynorsk | no_nynorsklia | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/no_nynorsklia_models.zip) |
| Old Church Slavonic | PROIEL | cu | cu_proiel | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/cu_proiel_models.zip) |
| Old French | SRCMF | fro | fro_srcmf | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/fro_srcmf_models.zip) |
| Persian | Seraji | fa | fa_seraji | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/fa_seraji_models.zip) |
| Polish | LFG | pl | pl_lfg | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/pl_lfg_models.zip) |
|  | SZ | pl | pl_sz | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/pl_sz_models.zip) |
| Portuguese | Bosque | pt | pt_bosque | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/pt_bosque_models.zip) |
| Romanian | RRT | ro | ro_rrt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ro_rrt_models.zip) |
| Russian | SynTagRus | ru | ru_syntagrus | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ru_syntagrus_models.zip) |
|  | Taiga | ru | ru_taiga | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ru_taiga_models.zip) |
| Serbian | SET | sr | sr_set | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/sr_set_models.zip) |
| Slovak | SNK | sk | sk_snk | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/sk_snk_models.zip) |
| Slovenian | SSJ | sl | sl_ssj | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/sl_ssj_models.zip) |
|  | SST | sl | sl_sst | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/sl_sst_models.zip) |
| Spanish | AnCora | es | es_ancora | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/es_ancora_models.zip) |
| Swedish | LinES | sv | sv_lines | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/sv_lines_models.zip) |
|  | Talbanken | sv | sv_talbanken | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/sv_talbanken_models.zip) |
| Turkish | IMST | tr | tr_imst | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/tr_imst_models.zip) |
| Ukrainian | IU | uk | uk_iu | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/uk_iu_models.zip) |
| Upper Sorbian | UFAL | hsb | hsb_ufal | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/hsb_ufal_models.zip) |
| Urdu | UDTB | ur | ur_udtb | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ur_udtb_models.zip) |
| Uyghur | UDT | ug | ug_udt | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/ug_udt_models.zip) |
| Vietnamese | VTB | vi | vi_vtb | [0.2.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/vi_vtb_models.zip) |

### 0.1.0

| Language | Treebank | Language code | Treebank code | Models |
| :------- | :------- | :------------ | :------------ | :----- |
| Afrikaans | AfriBooms | af | af_afribooms | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/af_afribooms_models.zip) |
| Ancient Greek | Perseus | grc | grc_perseus | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/grc_perseus_models.zip) |
|  | PROIEL | grc | grc_proiel | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/grc_proiel_models.zip) |
| Arabic | PADT | ar | ar_padt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ar_padt_models.zip) |
| Armenian | ArmTDP | hy | hy_armtdp | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/hy_armtdp_models.zip) |
| Basque | BDT | eu | eu_bdt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/eu_bdt_models.zip) |
| Bulgarian | BTB | bg | bg_btb | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/bg_btb_models.zip) |
| Buryat | BDT | bxr | bxr_bdt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/bxr_bdt_models.zip) |
| Catalan | AnCora | ca | ca_ancora | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ca_ancora_models.zip) |
| Chinese (traditional) | GSD | zh | zh_gsd | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/zh_gsd_models.zip) |
| Croatian | SET | hr | hr_set | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/hr_set_models.zip) |
| Czech | CAC | cs | cs_cac | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/cs_cac_models.zip) |
|  | FicTree | cs | cs_fictree | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/cs_fictree_models.zip) |
|  | PDT | cs | cs_pdt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/cs_pdt_models.zip) |
| Danish | DDT | da | da_ddt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/da_ddt_models.zip) |
| Dutch | Alpino | nl | nl_alpino | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/nl_alpino_models.zip) |
|  | LassySmall | nl | nl_lassysmall | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/nl_lassysmall_models.zip) |
| English | EWT | en | en_ewt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/en_ewt_models.zip) |
|  | GUM | en | en_gum | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/en_gum_models.zip) |
|  | LinES | en | en_lines | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/en_lines_models.zip) |
| Estonian | EDT | et | et_edt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/et_edt_models.zip) |
| Finnish | FTB | fi | fi_ftb | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/fi_ftb_models.zip) |
|  | TDT | fi | fi_tdt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/fi_tdt_models.zip) |
| French | GSD | fr | fr_gsd | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/fr_gsd_models.zip) |
|  | Sequoia | fr | fr_sequoia | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/fr_sequoia_models.zip) |
|  | Spoken | fr | fr_spoken | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/fr_spoken_models.zip) |
| Galician | CTG | gl | gl_ctg | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/gl_ctg_models.zip) |
|  | TreeGal | gl | gl_treegal | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/gl_treegal_models.zip) |
| German | GSD | de | de_gsd | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/de_gsd_models.zip) |
| Gothic | PROIEL | got | got_proiel | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/got_proiel_models.zip) |
| Greek | GDT | el | el_gdt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/el_gdt_models.zip) |
| Hebrew | HTB | he | he_htb | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/he_htb_models.zip) |
| Hindi | HDTB | hi | hi_hdtb | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/hi_hdtb_models.zip) |
| Hungarian | Szeged | hu | hu_szeged | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/hu_szeged_models.zip) |
| Indonesian | GSD | id | id_gsd | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/id_gsd_models.zip) |
| Irish | IDT | ga | ga_idt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ga_idt_models.zip) |
| Italian | ISDT | it | it_isdt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/it_isdt_models.zip) |
|  | PoSTWITA | it | it_postwita | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/it_postwita_models.zip) |
| Japanese | GSD | ja | ja_gsd | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ja_gsd_models.zip) |
| Kazakh | KTB | kk | kk_ktb | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/kk_ktb_models.zip) |
| Korean | GSD | ko | ko_gsd | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ko_gsd_models.zip) |
|  | Kaist | ko | ko_kaist | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ko_kaist_models.zip) |
| Kurmanji | MG | kmr | kmr_mg | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/kmr_mg_models.zip) |
| Latin | ITTB | la | la_ittb | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/la_ittb_models.zip) |
|  | Perseus | la | la_perseus | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/la_perseus_models.zip) |
|  | PROIEL | la | la_proiel | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/la_proiel_models.zip) |
| Latvian | LVTB | lv | lv_lvtb | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/lv_lvtb_models.zip) |
| North Sami | Giella | sme | sme_giella | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/sme_giella_models.zip) |
| Norwegian | Bokmaal | no_bokmaal | no_bokmaal | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/no_bokmaal_models.zip) |
|  | Nynorsk | no_nynorsk | no_nynorsk | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/no_nynorsk_models.zip) |
|  | NynorskLIA | no_nynorsk | no_nynorsklia | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/no_nynorsklia_models.zip) |
| Old Church Slavonic | PROIEL | cu | cu_proiel | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/cu_proiel_models.zip) |
| Old French | SRCMF | fro | fro_srcmf | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/fro_srcmf_models.zip) |
| Persian | Seraji | fa | fa_seraji | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/fa_seraji_models.zip) |
| Polish | LFG | pl | pl_lfg | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/pl_lfg_models.zip) |
|  | SZ | pl | pl_sz | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/pl_sz_models.zip) |
| Portuguese | Bosque | pt | pt_bosque | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/pt_bosque_models.zip) |
| Romanian | RRT | ro | ro_rrt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ro_rrt_models.zip) |
| Russian | SynTagRus | ru | ru_syntagrus | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ru_syntagrus_models.zip) |
|  | Taiga | ru | ru_taiga | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ru_taiga_models.zip) |
| Serbian | SET | sr | sr_set | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/sr_set_models.zip) |
| Slovak | SNK | sk | sk_snk | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/sk_snk_models.zip) |
| Slovenian | SSJ | sl | sl_ssj | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/sl_ssj_models.zip) |
|  | SST | sl | sl_sst | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/sl_sst_models.zip) |
| Spanish | AnCora | es | es_ancora | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/es_ancora_models.zip) |
| Swedish | LinES | sv | sv_lines | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/sv_lines_models.zip) |
|  | Talbanken | sv | sv_talbanken | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/sv_talbanken_models.zip) |
| Turkish | IMST | tr | tr_imst | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/tr_imst_models.zip) |
| Ukrainian | IU | uk | uk_iu | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/uk_iu_models.zip) |
| Upper Sorbian | UFAL | hsb | hsb_ufal | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/hsb_ufal_models.zip) |
| Urdu | UDTB | ur | ur_udtb | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ur_udtb_models.zip) |
| Uyghur | UDT | ug | ug_udt | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/ug_udt_models.zip) |
| Vietnamese | VTB | vi | vi_vtb | [0.1.0](http://nlp.stanford.edu/software/stanfordnlp_models/0.1.0/vi_vtb_models.zip) |
