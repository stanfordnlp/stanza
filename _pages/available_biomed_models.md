---
layout: page
title: Available Biomedical & Clinical Models
keywords: stanza, available biomedical and clinical models
permalink: '/available_biomed_models.html'
nav_order: 1
parent: Biomedical Models
---

At a high level, Stanza currently provides packages that support Universal Dependencies (UD)-compatible **syntactic analysis** and **named entity recognition (NER)** from both English biomedical literature and clinical note text. Officially offered packages include:
- 2 UD-compatible biomedical syntactic analysis pipelines, trained with human-annotated treebanks;
- 1 UD-compatible clinical syntactic analysis pipeline, trained with silver data;
- 8 accurate biomedical NER models augmented with contextualized representations;
- 2 clinical NER models, including one specialized in radiology reports.

All our syntactic analysis pipelines are compatible with the [Universal Dependencies v2 framework](https://universaldependencies.org/). Here we briefly introduce our models and their usage; for full details on the creation of these models and their evaluation, please check out [our biomedical models description paper](https://arxiv.org/abs/2007.14640).

⚒️ &nbsp;For a quick introduction on how to download and use these models, please visit the [Biomedical Models Download & Usage page](biomed_model_usage).

## Biomedical & Clinical Syntactic Analysis Pipelines

The following table lists the syntactic analysis pipelines currently offered in Stanza. You can find more information about the [POS tags](https://universaldependencies.org/u/pos/all.html), [morphological features](https://universaldependencies.org/u/feat/all.html), and [syntactic relations](https://universaldependencies.org/u/dep/all.html) used on the [Universal Dependencies website](https://universaldependencies.org/).

**Table Notes**

1. Package Name: this column lists the "keyword" that's needed to download this model and construct the pipeline. See usage for more details.
2. New Tokenization: this column lists whether the pipeline is compatible with the new UD-friendly tokenization or not. The most notable difference is that in new tokenization, hyphenated words are split into multiple tokens (e.g., `up-regulation` will be tokenized into three parts `up - regulation`); whereas in old tokenization hyphenated words are kept as a whole.
3. Source Corpora: this column describes the genres and domains of text that the training corpora of a particular model have.

| Category | Treebank | Package Name | New Tokenization | Source Corpora | Treebank Doc |
| :---- | :----- | :---- | :---- | :---- | :---- |
| Bio | CRAFT | craft | Yes | Full-text biomedical articles related to the Mouse Genome Informatics database; general English Web Treebank. | [CRAFT homepage](http://bionlp-corpora.sourceforge.net/CRAFT/) |
| | GENIA | genia | No | PubMed abstracts related to "human", "blood cells", and "transcription factors". | [GENIA homepage](http://www.geniaproject.org/)
| Clinical | MIMIC | mimic | Yes | All types of MIMIC-III clinical notes; general English Web Treebank. | [MIMIC-III homepage](https://mimic.physionet.org/) |


## Biomedical & Clinical NER Models

The following table lists all biomedical and clinical NER models supported by Stanza, pretrained on the corresponding NER datasets. 

| Category | Corpus | Package Name | Supported Entity Types |
| :---- | :---- | :---- | :---- |
| Bio | AnatEM | anatem | `ANATOMY` |
| | BC5CDR | bc5cdr | `CHEMICAL`, `DISEASE` |
| | BC4CHEMD | bc4chemd | `CHEMICAL` |
| | BioNLP13CG | bionlp13cg | 16 types in Cancer Genetics (* see below for a full list) |
| | JNLPBA | jnlpba | `PROTEIN`, `DNA`, `RNA`, `CELL_LINE`, `CELL_TYPE` |
| | Linnaeus | linnaeus | `SPECIES` |
| | NCBI-Disease | ncbi_disease | `DISEASE` |
| | S800 | s800 | `SPECIES` |
| Clinical | i2b2-2010 | i2b2 | `PROBLEM`, `TEST`, `TREATMENT` |
| | Radiology | radiology | `ANATOMY`, `OBSERVATION`, `ANATOMY_MODIFIER`, `OBSERVATION_MODIFIER`, `UNCERTAINTY` |

* The 16 entity types in the BioNLP13CG model include: `AMINO_ACID`, `ANATOMICAL_SYSTEM`, `CANCER`, `CELL`, `CELLULAR_COMPONENT`, `DEVELOPING_ANATOMICAL_STRUCTURE`, `GENE_OR_GENE_PRODUCT`, `IMMATERIAL_ANATOMICAL_ENTITY`, `MULTI-TISSUE_STRUCTURE`, `ORGAN`, `ORGANISM`, `ORGANISM_SUBDIVISION`, `ORGANISM_SUBSTANCE`, `PATHOLOGICAL_FORMATION`, `SIMPLE_CHEMICAL`, `TISSUE`.


Besides these datasets, all NER models in Stanza are augmented with pretrained character-level language models for improved accuracy. For the bio NER models, the language models are pretrained on the [publicly available PubMed abstracts](ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline); for the clinical NER models, the language models are pretrained on the clinical notes from the [publicly available MIMIC-III database](https://mimic.physionet.org/).
