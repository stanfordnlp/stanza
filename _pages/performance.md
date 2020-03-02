---
title: System Performance
keywords: stanfordnlp, system performance
permalink: '/performance.html'
---

## System Performance on UD Treebanks

In the table below you can find the performance of the version 1.0.0 models of StanfordNLP's neural pipeline, which is our best attempt at replicating our final system in the full PyTorch pipeline. The scores shown are from an end-to-end evaluation on the official test sets (from raw text to the full CoNLL-U file), and the scores are generated with the official evaluation script. For how we handled treebanks with no training data, please refer to the [system description paper]() for details. 

| Treebank | Tokens | Sentences | Words | UPOS | XPOS | UFeats | AllTags | Lemmas | UAS | LAS | CLAS | MLAS | BLEX |
| :------- | :----- | :-------- | :---- | :--- | :--- | :----- | :------ | :----- | :-- | :-- | :--- | :--- | :--- |
| Macro Avg | 98.64 | 86.81 | 98.43 | 90.99 | 87.88 | 87.71 | 82.03 | 90.51 | 79.87 | 74.82 | 70.91 | 63.26 | 66.04 | 
{: .compact #conll18-results }

## System Performance on NER Corpora

In the table below you can find the performance of the version 1.0.0 models of StanfordNLP's neural pipeline on various NER corpora.

| Language | Corpus | # Types | F1 |
| :------- | :----- | :-------- | :---- |
| Arabic | AQMAR | 4 | 74.3 |
| Chinese | OntoNotes | 18 | 79.2 |
| Dutch | CoNLL02 | 4 | 89.2 |
| Dutch | WikiNER | 4 | 94.8 |
| English | CoNLL03 | 4 | 92.1 |
| English | OntoNotes | 18 | 88.8 |
| French | WikiNER | 4 | 92.9 |
| German | CoNLL03 | 4 | 81.9 |
| German | GermEval14 | 4 | 85.2 |
| Russian | WikiNER | 4 | 92.9 |
| Spanish | CoNLL02 | 4 | 88.1 |
| Spanish | AnCora | 4 | 88.6 |
