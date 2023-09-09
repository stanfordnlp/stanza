---
layout: page
title: NER Models
keywords: ner models
permalink: '/ner_models.html'
nav_order: 5
parent: Models
datatable: true
---


## System Performance on NER Corpora

In the table below you can find the performance of Stanza's pretrained
NER models. All numbers reported are micro-averaged F1 scores. We used
canonical train/dev/test splits for all datasets except for the
WikiNER datasets, for which we used random splits.

The Ukrainian model and its score [was provided by gawy](https://github.com/stanfordnlp/stanza/issues/319).  The Armenian model [was provided by ShakeHakobyan](https://github.com/stanfordnlp/stanza/issues/1206).  The Polish model [was provided by Karol Saputa](https://github.com/stanfordnlp/stanza/issues/1070)

| Language          | lcode  | Corpus          | Types     | F1    | Def?                                               | Since                              |  Doc     |
| :--------------   | :----  | :-----          | :-------  | :---- | :------------------------------------------------: | :---:                              | :---------  |
| Afrikaans         |   af   | NCHLT           | 4         | 80.08 | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://hdl.handle.net/20.500.12185/299) |
| Armenian          |   hy   | ARMTDP          | 18        | 87.96 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.5.0</b> | [<i class="fas fa-file-alt"></i>](https://github.com/myavrum/ArmTDP-NER.git) |
| Arabic            |   ar   | AQMAR           | 4         | 74.3  | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](http://www.cs.cmu.edu/~ark/ArabicNER/) |
| Bulgarian         |   bg   | BSNLP 2019      | 5         | 83.21 | <i class="fas fa-check" style="color:#33a02c"></i> | 1.2.1                              | [<i class="fas fa-file-alt"></i>](http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html) |
| Chinese           |   zh   | OntoNotes       | 18        | 79.2  | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://catalog.ldc.upenn.edu/LDC2013T19) |
| Danish            |   da   | DDT             | 4         | 80.95 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.0</b> | [<i class="fas fa-file-alt"></i>](https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html#dane)  [<i class="fas fa-file-alt"></i>](https://aclanthology.org/2020.lrec-1.565.pdf)  |
| Dutch             |   nl   | CoNLL02         | 4         | 89.2  | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://www.aclweb.org/anthology/W02-2024.pdf) |
| Dutch             |   nl   | WikiNER         | 4         | 94.8  | <i class="fas fa-minus" style="color:#a0332c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500) |
| English           |   en   | CoNLL03         | 4         | 92.1  | <i class="fas fa-minus" style="color:#a0332c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://dl.acm.org/citation.cfm?id=1119195) |
| English           |   en   | OntoNotes       | 18        | 88.8  | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://catalog.ldc.upenn.edu/LDC2013T19) |
| Finnish           |   fi   | Turku           | 6         | 87.04 | <i class="fas fa-check" style="color:#33a02c"></i> | 1.2.1                              | [<i class="fas fa-file-alt"></i>](https://turkunlp.org/fin-ner.html) |
| French            |   fr   | WikiNER         | 4         | 92.9  | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500) |
| German            |   de   | CoNLL03         | 4         | 81.9  | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://dl.acm.org/citation.cfm?id=1119195) |
| German            |   de   | GermEval2014    | 4         | 85.2  | <i class="fas fa-minus" style="color:#a0332c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://sites.google.com/site/germeval2014ner/data) |
| Hungarian         |   hu   | Combined        | 4         | -     | <i class="fas fa-check" style="color:#33a02c"></i> | 1.2.1                              | [<i class="fas fa-file-alt"></i>](https://rgai.inf.u-szeged.hu/node/130)  [<i class="fas fa-file-alt"></i>](https://github.com/nytud/NYTK-NerKor) |
| Italian           |   it   | FBK             | 3         | 87.92 | <i class="fas fa-check" style="color:#33a02c"></i> | 1.2.3                              | [<i class="fas fa-file-alt"></i>](https://dh.fbk.eu/) |
| Japanese          |   ja   | GSD             | 22        | 81.01 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.0</b> | [<i class="fas fa-file-alt"></i>](https://github.com/megagonlabs/UD_Japanese-GSD) |
| Kazakh            |   kk   | kazNERD         | 25        | 94.94 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.1</b> | [<i class="fas fa-file-alt"></i>](https://arxiv.org/abs/2111.13419) [<i class="fas fa-file-alt"></i>](https://github.com/IS2AI/KazNERD) |
| Marathi           |   mr   | L3Cube          | 6         | 84.19 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.1</b> | [<i class="fas fa-file-alt"></i>](https://arxiv.org/abs/2205.14728) [<i class="fas fa-file-alt"></i>](https://aclanthology.org/2022.wildre-1.6.pdf)  [<i class="fas fa-file-alt"></i>](https://github.com/l3cube-pune/MarathiNLP) |
| Myanmar           |   my   | UCSY            | 7         | 95.86 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.0</b> | [<i class="fas fa-file-alt"></i>](https://arxiv.org/ftp/arxiv/papers/1903/1903.04739.pdf) |
| Norwegian&#8209;Bokmaal |   nb   | Norne           | 8         | 84.79 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.0</b> | [<i class="fas fa-file-alt"></i>](https://github.com/ltgoslo/norne) |
| Norwegian&#8209;Nynorsk |   nn   | Norne           | 8         | 80.16 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.0</b> | [<i class="fas fa-file-alt"></i>](https://github.com/ltgoslo/norne) |
| Persian           |   fa   | Arman           | 6         | 80.07 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.0</b> | [<i class="fas fa-file-alt"></i>](https://github.com/HaniehP/PersianNER) |
| Polish            |   pl   | NKJP            | 6         | 88.73 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.1</b> | [<i class="fas fa-file-alt"></i>](http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=view&target=NKJP-PodkorpusMilionowy-1.2.tar.gz) |
| Russian           |   ru   | WikiNER         | 4         | 92.9  | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500) |
| Sindhi            |   sd   | SiNER           | 11        | 84.74 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.5.0</b> | [<i class="fas fa-file-alt"></i>](https://github.com/AliWazir/SiNER-dataset) [<i class="fas fa-file-alt"></i>](https://aclanthology.org/2020.lrec-1.361/) |
| Spanish           |   es   | CoNLL02         | 4         | 88.1  | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://www.aclweb.org/anthology/W02-2024.pdf) |
| Spanish           |   es   | AnCora          | 4         | 88.6  | <i class="fas fa-minus" style="color:#a0332c"></i> |                                    | [<i class="fas fa-file-alt"></i>](http://clic.ub.edu/corpus/en) |
| Swedish           |   sv   | SUC3 (shuffled) | 8         | 85.66 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.0</b> | [<i class="fas fa-file-alt"></i>](https://spraakbanken.gu.se/en/resources/suc3) |
| Swedish           |   sv   | SUC3 (licensed) | 8         | 82.54 | <i class="fas fa-minus" style="color:#a0332c"></i> | <b style="color:#33a02c">1.4.0</b> | [<i class="fas fa-file-alt"></i>](https://spraakbanken.gu.se/en/resources/suc3) |
| Thai              |   th   | LST20           | 10        | 79.65 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.1</b> | [<i class="fas fa-file-alt"></i>](https://arxiv.org/ftp/arxiv/papers/2008/2008.05055.pdf) |
| Turkish           |   tr   | Starlang        | 5         | 81.65 | <i class="fas fa-check" style="color:#33a02c"></i> | <b style="color:#33a02c">1.4.0</b> | [<i class="fas fa-file-alt"></i>](https://ieeexplore.ieee.org/document/9259873) |
| Ukrainian         |   uk   | languk          | 4         | 86.05 | <i class="fas fa-check" style="color:#33a02c"></i> |                                    | [<i class="fas fa-file-alt"></i>](https://github.com/lang-uk/ner-uk) [<i class="fas fa-file-alt"></i>](https://github.com/gawy/stanza-lang-uk/releases/tag/v0.9)  |
| Vietnamese        |   vi   | VLSP            | 4         | 82.44 | <i class="fas fa-check" style="color:#33a02c"></i> | 1.2.1                              | [<i class="fas fa-file-alt"></i>](https://vlsp.org.vn/vlsp2018/eval/ner) |
{: .compact #ner-results .plain-datatable }

### Notes on NER Corpora

We have provided links to all NER datasets used to train the released models on our [available NER models page](available_models.md#available-ner-models). Here we provide notes on how to find several of these corpora:

- **Afrikaans**: The Afrikaans data is part of [the NCHLT corpus of South African languages](https://repo.sadilar.org/handle/20.500.12185/299).  Van Huyssteen, G.B., Puttkammer, M.J., Trollip, E.B., Liversage, J.C., Eiselen, R. 2016. [NCHLT Afrikaans Named Entity Annotated Corpus. 1.0](https://hdl.handle.net/20.500.12185/299).


- **Bulgarian**: The Bulgarian BSNLP 2019 data is available from [the shared task page](http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html). You can also find their [dataset description paper](https://www.aclweb.org/anthology/W19-3709/).

- **Finnish**: The Turku dataset used for Finnish NER training can be found on [the Turku NLP website](https://turkunlp.org/fin-ner.html), and they also provide [a Turku NER dataset description paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.567.pdf).

- **Hungarian**: The dataset used for training our Hungarian NER system is a combination of 3 separate datasets: Business, Criminal, and NYTK. Two of these datasets can be found from [this Szeged page](https://rgai.inf.u-szeged.hu/node/130), and the third can be found in [this NYTK-NerKor github repo](https://github.com/nytud/NYTK-NerKor). A dataset description paper can also be found [here](http://www.inf.u-szeged.hu/projectdirs/hlt/papers/lrec_ne-corpus.pdf).

- **Italian**: The Italian FBK dataset was licensed to us from [FBK](https://dh.fbk.eu/).  Paccosi T. and Palmero Aprosio A.  KIND: an Italian Multi-Domain Dataset for Named Entity Recognition.  LREC 2022

- **Marathi**: The L3cube-MahaNER dataset was used for Marathi NER training. The original dataset was curated as a part of the [MahaNLP](https://arxiv.org/abs/2205.14728) initiative by L3Cube Pune. The MahaNER along with other Marathi resources is shared on the [Marathi NLP](https://github.com/l3cube-pune/MarathiNLP) page and described in their [dataset description paper](https://aclanthology.org/2022.wildre-1.6.pdf).

- **Myanmar**: The Myanmar dataset is by special request from [UCSY](https://arxiv.org/ftp/arxiv/papers/1903/1903.04739.pdf).

- **Swedish**: The [SUC3 dataset](https://spraakbanken.gu.se/en/resources/suc3) has two versions, one with the entries shuffled and another using the original ordering of the data.  We make the shuffled version the default in order to expand the coverage of the model.

- **Vietnamese**: The Vietnamese VLSP dataset is available by [request from VLSP](https://vlsp.org.vn/vlsp2018/eval/ner).

**Tag category notes**

- For packages with 4 named entity types, supported types include `PER` (Person), `LOC` (Location), `ORG` (Organization) and `MISC` (Miscellaneous)
    - The Vietnamese VLSP model spells out the entire tag, though: PERSON, LOCATION, ORGANIZATION, MISCELLANEOUS.
- For packages with 18 named entity types, supported types include `PERSON`, `NORP` (Nationalities/religious/political group), `FAC` (Facility), `ORG` (Organization), `GPE` (Countries/cities/states), `LOC` (Location), `PRODUCT`,`EVENT`, `WORK_OF_ART`, `LAW`, `LANGUAGE`, `DATE`, `TIME`, `PERCENT`, `MONEY`, `QUANTITY`, `ORDINAL` and `CARDINAL` (details can be found on page 21 of this [OntoNotes documentation](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf)).
- The BSNLP19 dataset(s) use `EVENT`, `LOCATION`, `ORGANIZATION`, `PERSON`, `PRODUCT`.
- The Italian FBK dataset uses `LOC`, `ORG`, `PER`
- The Marathi L3Cube dataset uses `ED` (Designation), `NED` (Date), `NEL` (Location), `NEM` (Measure), `NEO` (Organization), `NEP` (Person), `NETI` (Time)
- The Myanmar UCSY dataset uses `LOC` (Location), `NE` (Misc), `ORG` (Organization), `PNAME` (Person), `RACE`, `TIME`, `NUM`
- The Japanese GSD dataset uses 22 tags: `CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `MOVEMENT`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PET_NAME`, `PHONE`, `PRODUCT`, `QUANTITY`, `TIME`, `TITLE_AFFIX`, `WORK_OF_ART`
- The Kazakh KazNERD dataset uses 25 tags: `ADAGE`, `ART`, `CARDINAL`, `CONTACT`, `DATE`, `DISEASE`, `EVENT`, `FACILITY`, `GPE`, `LANGUAGE`, `LAW`, `LOCATION`, `MISCELLANEOUS`, `MONEY`, `NON_HUMAN`, `NORP`, `ORDINAL`, `ORGANISATION`, `PERCENTAGE`, `PERSON`, `POSITION`, `PRODUCT`, `PROJECT`, `QUANTITY`, `TIME`
- The Norwegian Norne dataset uses 8 tags for both NB and NN: `DRV`, `EVT`, `GPE`, `LOC`, `MISC`, `ORG`, `PER`, `PROD`
- The Persian Arman dataset uses 6 tags: `event`, `fac`, `loc`, `org`, `pers`, `pro`
- The Polish NKJP dataset uses 6 tags: `date`, `geogName`, `orgName`, `persName`, `placeName`, `time`
- The Sindhi SiNER dataset uses 11 tags: `ART`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LOC`, `NORP`, `ORG`, `OTHERS`, `PERSON`, `TITLE`
- The Thai LST20 dataset uses 10 tags: Person (`PER`), Title (`TTL`), Designator (`DES`), Organization (`ORG`), Location (`LOC`), Brand (`BRN`), Date and time (`DTM`), Measurement unit (`MEA`), Number (`NUM`), and Terminology (`TRM`)
- The Turkish Starlang dataset uses 5 tags: `LOCATION`, `MONEY`, `ORGANIZATION`, `PERSON`, `TIME`
