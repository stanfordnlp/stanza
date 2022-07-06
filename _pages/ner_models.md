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

The Ukrainian model and its score [was provided by a user](https://github.com/stanfordnlp/stanza/issues/319).

| Language                         | LCODE  | Corpus          | # Types   | F1 |
| :--------------------            | :----  | :-----          | :-------- | :---- |
| Afrikaans                        |   af   | NCHLT           | 4         | 80.08 |
| Arabic                           |   ar   | AQMAR           | 4         | 74.3  |
| Bulgarian *New in 1.2.1*         |   bg   | BSNLP 2019      | 5         | 83.21 |
| Chinese                          |   zh   | OntoNotes       | 18        | 79.2  |
| Danish *New in 1.4.0*            |   da   | DDT             | 4         | 80.95 |
| Dutch                            |   nl   | CoNLL02         | 4         | 89.2  |
| Dutch                            |   nl   | WikiNER         | 4         | 94.8  |
| English                          |   en   | CoNLL03         | 4         | 92.1  |
| English                          |   en   | OntoNotes       | 18        | 88.8  |
| Finnish *New in 1.2.1*           |   fi   | Turku           | 6         | 87.04 |
| French                           |   fr   | WikiNER         | 4         | 92.9  |
| German                           |   de   | CoNLL03         | 4         | 81.9  |
| German                           |   de   | GermEval2014    | 4         | 85.2  |
| Hungarian *New in 1.2.1*         |   hu   | Combined        | 4         | -     |
| Italian *New in 1.2.3*           |   it   | FBK             | 3         | 87.92 |
| Japanese *New in 1.4.0*          |   ja   | GSD             | 22        | 81.01 |
| Myanmar *New in 1.4.0*           |   my   | UCSY            | 7         | 95.86 |
| Norwegian-Bokmaal *New in 1.4.0* |   nb   | Norne           | 8         | 84.79 |
| Norwegian-Nynorsk *New in 1.4.0* |   nn   | Norne           | 8         | 80.16 |
| Persian *New in 1.4.0*           |   fa   | Arman           | 6         | 80.07 |
| Russian                          |   ru   | WikiNER         | 4         | 92.9  |
| Spanish                          |   es   | CoNLL02         | 4         | 88.1  |
| Spanish                          |   es   | AnCora          | 4         | 88.6  |
| Swedish *New in 1.4.0*           |   sv   | SUC3 (shuffled) | 8         | 85.66 |
| Swedish *New in 1.4.0*           |   sv   | SUC3 (licensed) | 8         | 82.54 |
| Turkish *New in 1.4.0*           |   tr   | Starlang        | 5         | 81.65 |
| Ukrainian                        |   uk   | languk          | 4         | 86.05 |
| Vietnamese *New in 1.2.1*        |   vi   | VLSP            | 4         | 82.44 |
{: .compact #ner-results .datatable }

### Notes on NER Corpora

We have provided links to all NER datasets used to train the released models on our [available NER models page](available_models.md#available-ner-models). Here we provide notes on how to find several of these corpora:

- **Afrikaans**: The Afrikaans data is part of [the NCHLT corpus of South African languages](https://repo.sadilar.org/handle/20.500.12185/299).  Van Huyssteen, G.B., Puttkammer, M.J., Trollip, E.B., Liversage, J.C., Eiselen, R. 2016. [NCHLT Afrikaans Named Entity Annotated Corpus. 1.0](https://hdl.handle.net/20.500.12185/299).


- **Bulgarian**: The Bulgarian BSNLP 2019 data is available from [the shared task page](http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html). You can also find their [dataset description paper](https://www.aclweb.org/anthology/W19-3709/).

- **Finnish**: The Turku dataset used for Finnish NER training can be found on [the Turku NLP website](https://turkunlp.org/fin-ner.html), and they also provide [a Turku NER dataset description paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.567.pdf).

- **Hungarian**: The dataset used for training our Hungarian NER system is a combination of 3 separate datasets. Two of these datasets can be found from [this Szeged page](https://rgai.inf.u-szeged.hu/node/130), and the third can be found in [this NYTK-NerKor github repo](https://github.com/nytud/NYTK-NerKor). A dataset description paper can also be found [here](http://www.inf.u-szeged.hu/projectdirs/hlt/papers/lrec_ne-corpus.pdf).

- **Italian**: The Italian FBK dataset was licensed to us from [FBK](https://dh.fbk.eu/).  Paccosi T. and Palmero Aprosio A.  KIND: an Italian Multi-Domain Dataset for Named Entity Recognition.  LREC 2022

- **Myanmar**: The Myanmar dataset is by special request from [UCSY](https://arxiv.org/ftp/arxiv/papers/1903/1903.04739.pdf).

- **Swedish**: The [SUC3 dataset] has two versions, one with the entries shuffled and another using the original ordering of the data.  We make the shuffled version the default in order to expand the coverage of the model.

- **Vietnamese**: The Vietnamese VLSP dataset is available by [request from VLSP](https://vlsp.org.vn/vlsp2018/eval/ner).

