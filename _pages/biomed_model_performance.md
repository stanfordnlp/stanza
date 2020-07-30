---
layout: page
title: Biomedical & Clinical Model Performance
keywords: stanza, biomedical and clinical model performance
permalink: '/biomed_model_performance.html'
nav_order: 2
parent: Biomedical Models
---


Here we report the performance of Stanza's biomedical and clinical models, including the syntactic analysis pipelines and the NER models. For more detailed evaluation and analysis, please see [our biomedical models description paper](https://arxiv.org/abs/2007.14640).

## Syntactic Analysis Performance

In the table below you can find the performance of Stanza's biomedical syntactic analysis pipelines. All models are evaluated on the test split of the corresponding datasets. Note that all scores reported are from an end-to-end evaluation on the official test sets (from raw text to the full CoNLL-U file with syntactic annotations), and are generated with the [CoNLL 2018 UD shared task](https://universaldependencies.org/conll18/evaluation.html) official evaluation script.

Note that while the results on the CRAFT and GENIA treebanks are based on human-annotated oracle data, the results on the MIMIC treebank is based on automatically generated silver data, and therefore should not be treated as a rigorous oracle-based evaluation.

| Category | Treebank | Package Name | Tokens | Sentences | UPOS | XPOS | Lemmas | UAS | LAS |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Bio | CRAFT | craft | 99.66 | 99.16 | 98.18 | 97.95 | 98.92 | 91.09 | 89.67  |
| | GENIA | genia | 99.81 | 99.78 | 98.81 | 98.76 | 99.58 | 91.01 | 89.48 |
| Clinical | MIMIC | mimic | 99.18 | 97.11 | 95.64 | 95.25 | 97.37 | 85.44 | 82.81 |

## NER Performance

In the table below you can find the performance of Stanza’s biomedical and clinical NER models, and their comparisons to the BioBERT models and scispaCy models. All numbers reported are micro-averaged F1 scores. We used canonical train/dev/test splits for all datasets, whenever such splits exist.

Results for BioBERT are from their v1.1 models as reported in the [BioBERT paper](https://arxiv.org/abs/1901.08746); results for scispaCy are from the medium sized models as reported in the [scispaCy paper](https://arxiv.org/abs/1902.07669).

| Category | Dataset | Domain & Types | Stanza | BioBERT | scispaCy |
| :------- | :----- | :-------- | :---- | :---- | :---- |
| Bio | AnatEM | Anatomy | 88.18 | -- | 84.14  |
| | BC5CDR | Chemical, Disease | 88.08 | -- | 83.92  |
| | BC4CHEMD | Chemical | 89.65 | 92.36 | 84.55  |
| | BioNLP13CG | 16 types in Cancer Genetics | 84.34 | -- | 77.60  |
| | JNLPBA | Protein, DNA, RNA, Cell line, Cell type | 76.09 | 77.49 | 73.21  |
| | Linnaeus | Species | 88.27 | 88.24 | 81.74  |
| | NCBI-Disease | Disease | 87.49 | 89.71 | 81.65  |
| | S800 | Species | 76.35 | 74.06 | --  |
| Clinical | i2b2 | Problem, Test, Treatment | 88.13 | 86.73 | -- |
| | Radiology | 5 types in Radiology | 84.80 | -- | --  |

