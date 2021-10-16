---
layout: page
title: Model Performance
keywords: stanza, system performance
permalink: '/performance.html'
nav_order: 2
parent: Models
datatable: true
---

Here we report the performance of Stanza's pretrained models on all supported languages. Again, performances of models for tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging and dependency parsing are reported on the Universal Dependencies (UD) treebanks, while performances of the NER models are reported separately.

## System Performance on UD Treebanks

In the table below you can find the performance of Stanza's pretrained UD models. All models are trained and tested with the Universal Dependencies v2.5 treebanks.
Note that all scores reported are from an end-to-end evaluation on the official test sets (from raw text to the full CoNLL-U file), and are generated with the CoNLL 2018 UD shared task official evaluation script. For detailed interpretation of these scores and the evaluation scripts we used, please refer to the [CoNLL 2018 UD Shared Task Evaluation](https://universaldependencies.org/conll18/evaluation.html) page. For details on how we handled treebanks with no training data, please refer to [our CoNLL 2018 system description paper](https://nlp.stanford.edu/pubs/qi2018universal.pdf).

For more detailed results and better viewing experience, please access the results in [this Google Spreadsheet](https://docs.google.com/spreadsheets/d/1t9h8QxjYA2XK4qs9Q7R4wiOwQHPykpBTjH8PGj_on5Y/edit?usp=sharing).

| Treebank | Tokens | Sentences | Words | UPOS | XPOS | UFeats | AllTags | Lemmas | UAS | LAS | CLAS | MLAS | BLEX |
| :------- | :----- | :-------- | :---- | :--- | :--- | :----- | :------ | :----- | :-- | :-- | :--- | :--- | :--- |
| Macro Avg | 98.64 | 86.81 | 98.43 | 90.99 | 87.88 | 87.71 | 82.03 | 90.51 | 79.87 | 74.82 | 70.91 | 63.26 | 66.04 |
| UD_Afrikaans-AfriBooms | 99.75 | 99.65 | 99.75 | 97.56 | 94.27 | 97.03 | 94.24 | 97.48 | 87.51 | 84.45 | 78.58 | 74.7 | 75.39 |
| UD_Ancient_Greek-Perseus | 99.98 | 98.85 | 99.98 | 92.54 | 85.22 | 91.06 | 84.98 | 88.26 | 78.75 | 73.35 | 67.88 | 54.22 | 57.54 |
| UD_Ancient_Greek-PROIEL | 100 | 51.65 | 100 | 97.38 | 97.75 | 92.09 | 90.96 | 97.42 | 80.34 | 76.33 | 71.37 | 61.23 | 69.23 |
| UD_Arabic-PADT | 99.98 | 80.43 | 97.88 | 94.89 | 91.75 | 91.86 | 91.51 | 93.27 | 83.27 | 79.33 | 76.24 | 70.58 | 72.79 |
| UD_Armenian-ArmTDP | 98.96 | 95.22 | 96.58 | 92.49 | 96.58 | 88.19 | 86.94 | 92.27 | 78.18 | 72.46 | 68.5 | 60.39 | 65.88 |
| UD_Basque-BDT | 100 | 100 | 100 | 96.23 | 100 | 93.09 | 91.34 | 96.52 | 86.19 | 82.76 | 81.29 | 73.56 | 78.26 |
| UD_Belarusian-HSE | 99.38 | 78.24 | 99.38 | 91.92 | 31.34 | 77.73 | 26.31 | 79.48 | 69.28 | 63.88 | 58.49 | 41.88 | 44.05 |
| UD_Bulgarian-BTB | 99.93 | 97.27 | 99.93 | 98.68 | 96.35 | 97.59 | 95.75 | 97.29 | 93.37 | 90.21 | 86.84 | 83.71 | 83.62 |
| UD_Buryat-BDT | 96.5 | 87.83 | 96.5 | 41.83 | 96.5 | 33.72 | 26.77 | 52.45 | 30.61 | 13.49 | 8.97 | 1.66 | 3.48 |
| UD_Catalan-AnCora | 99.99 | 99.84 | 99.98 | 98.75 | 98.66 | 98.29 | 97.74 | 98.47 | 92.84 | 90.56 | 86.25 | 84.07 | 85.31 |
| UD_Chinese-GSD | 92.83 | 98.8 | 92.83 | 89.12 | 88.93 | 92.11 | 88.18 | 92.83 | 72.88 | 69.82 | 66.81 | 63.26 | 66.81 |
| UD_Chinese-GSDSimp | 92.92 | 99.1 | 92.92 | 89.05 | 88.84 | 92.12 | 88.03 | 92.92 | 73.44 | 70.44 | 67.69 | 64.07 | 67.69 |
| UD_Classical_Chinese-Kyoto | 99.47 | 46.95 | 99.47 | 90.25 | 89.64 | 92.68 | 87.34 | 99.45 | 71.81 | 66.08 | 64.54 | 62.61 | 64.54 |
| UD_Coptic-Scriptorium | 99.94 | 29.99 | 86.06 | 82.37 | 80.25 | 73.51 | 69.47 | 83.12 | 61.94 | 59.71 | 50.49 | 34.17 | 49.54 |
| UD_Croatian-SET | 99.96 | 98.15 | 99.96 | 97.88 | 94.86 | 95.32 | 94.22 | 96.67 | 90.27 | 85.56 | 82.43 | 76.37 | 78.78 |
| UD_Czech-CAC | 99.99 | 100 | 99.97 | 98.76 | 94.79 | 93.52 | 92.65 | 98 | 91.7 | 89.19 | 86.84 | 80.14 | 84.89 |
| UD_Czech-CLTT | 99.93 | 100 | 99.84 | 98.92 | 91.89 | 91.97 | 91.28 | 97.48 | 86.67 | 83.38 | 79.35 | 70.7 | 77.56 |
| UD_Czech-FicTree | 99.97 | 98.6 | 99.96 | 98.31 | 95.23 | 96.01 | 94.58 | 98.43 | 92.69 | 89.81 | 87.3 | 81.94 | 85.42 |
| UD_Czech-PDT | 99.97 | 94.14 | 99.97 | 98.5 | 95.38 | 94.61 | 93.67 | 98.55 | 91 | 88.64 | 86.91 | 81.12 | 85.45 |
| UD_Danish-DDT | 99.96 | 93.57 | 99.96 | 97.75 | 99.96 | 97.38 | 96.45 | 97.32 | 86.83 | 84.19 | 81.2 | 77.13 | 78.46 |
| UD_Dutch-Alpino | 99.96 | 89.98 | 99.96 | 96.33 | 94.76 | 96.28 | 94.13 | 96.97 | 89.56 | 86.44 | 81.22 | 75.76 | 77.8 |
| UD_Dutch-LassySmall | 99.9 | 77.95 | 99.9 | 95.97 | 94.87 | 96.22 | 94.05 | 97.59 | 85.34 | 81.93 | 75.54 | 71.98 | 73.49 |
| UD_English-EWT | 99.01 | 81.13 | 99.01 | 95.4 | 95.12 | 96.11 | 93.9 | 97.21 | 86.22 | 83.59 | 80.21 | 76.02 | 78.5 |
| UD_English-GUM | 99.82 | 86.35 | 99.82 | 95.89 | 95.91 | 96.87 | 94.99 | 96.8 | 87.06 | 83.57 | 78.42 | 74.68 | 74.97 |
| UD_English-LinES | 99.95 | 88.49 | 99.95 | 96.88 | 95.18 | 96.76 | 93.11 | 98.32 | 85.82 | 81.97 | 79.04 | 74.47 | 77.31 |
| UD_English-ParTUT | 99.68 | 100 | 99.59 | 96.15 | 95.83 | 95.21 | 93.92 | 97.45 | 90.31 | 87.35 | 82.56 | 76.19 | 80.53 |
| UD_Estonian-EDT | 99.96 | 93.32 | 99.96 | 97.19 | 98.04 | 95.77 | 94.43 | 96.05 | 86.68 | 83.82 | 82.41 | 77.63 | 78.32 |
| UD_Estonian-EWT | 99.2 | 67.14 | 99.2 | 88.86 | 91.7 | 87.16 | 83.43 | 85.62 | 67.23 | 60.07 | 56.21 | 48.32 | 47.38 |
| UD_Finnish-FTB | 100 | 89.59 | 99.97 | 95.5 | 95.12 | 96.51 | 93.92 | 96.16 | 89.09 | 86.39 | 83.8 | 79.9 | 81.02 |
| UD_Finnish-TDT | 99.77 | 93.05 | 99.73 | 96.97 | 97.72 | 95.36 | 94.44 | 94.98 | 88.62 | 86.18 | 84.66 | 79.73 | 80.24 |
| UD_French-GSD | 99.68 | 94.92 | 99.48 | 97.3 | 99.47 | 96.72 | 96.05 | 97.64 | 91.38 | 89.05 | 84.38 | 80.3 | 82.4 |
| UD_French-ParTUT | 99.82 | 100 | 99.37 | 96.6 | 96.37 | 93.98 | 93.41 | 95.48 | 90.71 | 88.37 | 83.37 | 74.41 | 77.88 |
| UD_French-Sequoia | 99.9 | 88.79 | 99.58 | 98.19 | 99.58 | 97.58 | 96.94 | 98.25 | 90.47 | 88.34 | 84.71 | 81.77 | 83.31 |
| UD_French-Spoken | 100 | 22.09 | 99.45 | 95.49 | 97.06 | 99.45 | 93.23 | 96.53 | 75.82 | 70.71 | 62.13 | 59.57 | 60.44 |
| UD_Galician-CTG | 99.89 | 99.13 | 99.32 | 97.21 | 96.99 | 99.14 | 96.71 | 97.94 | 85.22 | 82.66 | 77.24 | 71.13 | 75.96 |
| UD_Galician-TreeGal | 99.59 | 89.17 | 98.41 | 94.29 | 91.81 | 93.36 | 90.88 | 94.39 | 78.04 | 72.94 | 65.61 | 59.06 | 61.49 |
| UD_German-GSD | 99.53 | 85.79 | 99.53 | 94.07 | 96.98 | 89.52 | 84.51 | 96.37 | 85.39 | 80.61 | 75.38 | 58.57 | 71.24 |
| UD_German-HDT | 100 | 97.41 | 100 | 98.04 | 97.94 | 91.77 | 91.34 | 97.48 | 94.91 | 92.59 | 88.73 | 77.26 | 85.63 |
| UD_Gothic-PROIEL | 100 | 39.71 | 100 | 96.17 | 96.71 | 90.62 | 88.86 | 96.48 | 74.67 | 69.03 | 66.21 | 57.11 | 64.38 |
| UD_Greek-GDT | 99.88 | 93.18 | 99.89 | 97.84 | 97.84 | 94.94 | 94.33 | 96.49 | 91.12 | 88.78 | 84.12 | 78 | 79.48 |
| UD_Hebrew-HTB | 99.98 | 99.69 | 93.19 | 90.46 | 90.46 | 89.24 | 88.45 | 90.27 | 79.18 | 76.6 | 71.05 | 64.51 | 67.79 |
| UD_Hindi-HDTB | 100 | 99.44 | 100 | 97.59 | 97.08 | 94.03 | 92.11 | 96.66 | 94.8 | 91.74 | 88.2 | 78.73 | 87.01 |
| UD_Hungarian-Szeged | 99.87 | 97 | 99.87 | 96.03 | 99.87 | 93.76 | 92.94 | 94.25 | 83.62 | 78.86 | 77.14 | 69.46 | 71.87 |
| UD_Indonesian-GSD | 99.99 | 93.78 | 99.99 | 93.68 | 94.79 | 96 | 89.17 | 99.61 | 85.17 | 79.19 | 77.04 | 68.86 | 76.68 |
| UD_Irish-IDT | 99.76 | 95.93 | 99.76 | 93.9 | 92.43 | 78.19 | 75 | 91.79 | 82.65 | 74.03 | 66.11 | 42.98 | 59.09 |
| UD_Italian-ISDT | 99.91 | 98.76 | 99.76 | 98.01 | 97.91 | 97.72 | 97.11 | 98.1 | 92.79 | 90.84 | 86.43 | 83.6 | 84.23 |
| UD_Italian-ParTUT | 99.81 | 100 | 99.77 | 97.82 | 97.76 | 97.79 | 96.94 | 97.57 | 92.24 | 90.01 | 84.39 | 81.77 | 82.05 |
| UD_Italian-PoSTWITA | 99.71 | 63.7 | 99.46 | 96.19 | 96.04 | 96.28 | 95.01 | 96.7 | 82.67 | 78.27 | 72.2 | 68.55 | 70.35 |
| UD_Italian-TWITTIRO | 99.34 | 52.4 | 98.76 | 94.41 | 94.01 | 93.34 | 91.45 | 93.17 | 78.87 | 72.85 | 64.64 | 58.67 | 59.35 |
| UD_Italian-VIT | 99.98 | 94.92 | 99.49 | 97.21 | 96.23 | 96.79 | 94.99 | 98.01 | 89.32 | 85.87 | 80.26 | 76.16 | 78.61 |
| UD_Japanese-GSD | 92.67 | 94.57 | 92.67 | 91.16 | 90.84 | 92.66 | 90.84 | 92.02 | 81.2 | 80.16 | 71.39 | 69.85 | 71.01 |
| UD_Kazakh-KTB | 93.46 | 88.56 | 94.16 | 56.23 | 56.1 | 42.73 | 36.96 | 52.12 | 44.33 | 25.21 | 20.28 | 7.63 | 10.01 |
| UD_Korean-GSD | 99.88 | 96.65 | 99.88 | 96.18 | 90.14 | 99.66 | 88 | 92.69 | 87.29 | 83.53 | 81.34 | 79.29 | 75.31 |
| UD_Korean-Kaist | 100 | 99.93 | 100 | 95.45 | 86.31 | 100 | 86.31 | 93.02 | 88.41 | 86.38 | 83.95 | 80.63 | 77.57 |
| UD_Kurmanji-MG | 94.81 | 87.43 | 94.49 | 57.17 | 55.91 | 43.02 | 38.41 | 56.13 | 32.01 | 21.91 | 16.35 | 3.84 | 5.84 |
| UD_Latin-ITTB | 99.99 | 80.66 | 99.99 | 98.09 | 95.38 | 96.43 | 93.8 | 98.9 | 87.61 | 85.36 | 84.23 | 80.28 | 83.6 |
| UD_Latin-Perseus | 100 | 98.24 | 100 | 90.63 | 78.42 | 82.42 | 77.74 | 83.08 | 71.94 | 61.99 | 57.89 | 45.28 | 47.28 |
| UD_Latin-PROIEL | 100 | 43.04 | 100 | 96.92 | 97.1 | 91.24 | 90.32 | 96.78 | 76.55 | 72.37 | 70.06 | 61.28 | 68.19 |
| UD_Latvian-LVTB | 99.82 | 99.01 | 99.82 | 96.03 | 88.25 | 93.46 | 87.73 | 95.55 | 87.84 | 84.44 | 82.16 | 73.91 | 78.25 |
| UD_Lithuanian-ALKSNIS | 99.87 | 88.79 | 99.87 | 93.37 | 85.67 | 87.84 | 84.84 | 92.51 | 78.54 | 73.11 | 70.66 | 60.81 | 65.53 |
| UD_Lithuanian-HSE | 97.53 | 51.11 | 97.53 | 81.08 | 80.04 | 70.72 | 66.44 | 76.9 | 48.1 | 37.45 | 32.37 | 21.1 | 24.86 |
| UD_Livvi-KKPP | 90.82 | 65.16 | 90.82 | 38.91 | 36.23 | 27.78 | 22.41 | 41.25 | 24.61 | 10.04 | 5.14 | 0.62 | 0.92 |
| UD_Maltese-MUDT | 99.86 | 84.94 | 99.86 | 95.75 | 95.63 | 99.86 | 95.31 | 99.86 | 83.31 | 78.15 | 70.64 | 67.15 | 70.64 |
| UD_Marathi-UFAL | 98 | 76.4 | 92.25 | 77.24 | 92.25 | 60.27 | 58.55 | 75.77 | 66.42 | 52.64 | 42.8 | 24.15 | 33.9 |
| UD_North_Sami-Giella | 99.68 | 99.31 | 99.68 | 91.11 | 92.85 | 87.72 | 83.8 | 88.79 | 74.22 | 68.43 | 65.59 | 58.32 | 58.13 |
| UD_Norwegian-Nynorsk | 99.97 | 94.85 | 99.97 | 97.92 | 99.97 | 96.88 | 96.03 | 97.9 | 91.87 | 89.73 | 87.28 | 82.86 | 84.78 |
| UD_Norwegian-NynorskLIA | 100 | 99.69 | 100 | 95.92 | 100 | 94.82 | 92.7 | 97.72 | 77.82 | 72.94 | 67.56 | 61.32 | 65.54 |
| UD_Norwegian-Bokmaal | 99.99 | 97.17 | 99.99 | 98.29 | 99.99 | 97.17 | 96.41 | 98.36 | 92.57 | 90.69 | 88.32 | 84.41 | 86.33 |
| UD_Old_Church_Slavonic-PROIEL | 100 | 49.26 | 100 | 96.58 | 96.88 | 90.65 | 89.63 | 95.69 | 79.75 | 74.93 | 74.64 | 65.45 | 72.02 |
| UD_Old_French-SRCMF | 100 | 100 | 100 | 96.05 | 96.09 | 97.74 | 95.56 | 100 | 91.38 | 86.35 | 83.39 | 80.05 | 83.39 |
| UD_Old_Russian-TOROT | 100 | 35.69 | 100 | 93.63 | 93.83 | 86.76 | 84.8 | 91.35 | 72.94 | 67 | 63.6 | 54.13 | 59.18 |
| UD_Persian-Seraji | 100 | 99.25 | 99.65 | 97.29 | 97.3 | 97.37 | 96.86 | 97.73 | 89.45 | 86.06 | 82.78 | 81 | 81.08 |
| UD_Polish-LFG | 99.95 | 99.83 | 99.95 | 98.55 | 94.66 | 95.84 | 94.07 | 96.86 | 95.8 | 93.94 | 92.35 | 87.62 | 88.64 |
| UD_Polish-PDB | 99.87 | 98.39 | 99.83 | 98.31 | 94.04 | 94.27 | 93.13 | 97.29 | 92.68 | 90.4 | 88.35 | 81.69 | 85.42 |
| UD_Portuguese-Bosque | 99.77 | 94.3 | 99.67 | 97.04 | 99.67 | 96.36 | 94.91 | 97.8 | 90.67 | 87.57 | 82.59 | 76.78 | 80.3 |
| UD_Portuguese-GSD | 99.96 | 98 | 99.87 | 98.18 | 98.18 | 99.79 | 98.17 | 95.83 | 92.83 | 91.36 | 87.44 | 85.87 | 86.75 |
| UD_Romanian-Nonstandard | 98.96 | 97.53 | 98.96 | 95.4 | 90.73 | 89.79 | 88.19 | 94.63 | 87.24 | 82.71 | 77.6 | 65.24 | 73.52 |
| UD_Romanian-RRT | 99.77 | 96.64 | 99.77 | 97.54 | 96.97 | 97.13 | 96.75 | 97.95 | 90.66 | 85.85 | 81.49 | 77.94 | 79.84 |
| UD_Russian-GSD | 99.65 | 97.16 | 99.65 | 97.38 | 97.18 | 93.11 | 92.22 | 95.34 | 88.97 | 84.83 | 82.37 | 75.16 | 77.75 |
| UD_Russian-SynTagRus | 99.57 | 98.86 | 99.57 | 98.2 | 99.57 | 95.91 | 95.59 | 97.51 | 92.38 | 90.6 | 89.01 | 85.04 | 86.78 |
| UD_Russian-Taiga | 97.11 | 85.79 | 97.11 | 92.25 | 94.7 | 85.76 | 82.61 | 89.28 | 72.09 | 66 | 61.8 | 51.94 | 55.64 |
| UD_Scottish_Gaelic-ARCOSG | 99.48 | 55.35 | 99.47 | 92.5 | 84.89 | 87.99 | 83.93 | 95.51 | 77.9 | 70.81 | 62.63 | 54 | 59.74 |
| UD_Serbian-SET | 100 | 99.33 | 100 | 98.44 | 94.26 | 94.55 | 93.86 | 96.34 | 91.79 | 88.78 | 86.5 | 79.48 | 82.38 |
| UD_Slovak-SNK | 99.97 | 90.93 | 99.97 | 96.34 | 87.15 | 91.59 | 86.34 | 94.73 | 89.96 | 86.82 | 84.74 | 75.39 | 79.35 |
| UD_Slovenian-SSJ | 99.91 | 91.6 | 99.91 | 98.29 | 95.08 | 95.37 | 94.56 | 97.34 | 91.63 | 89.6 | 87.18 | 82.35 | 84.37 |
| UD_Slovenian-SST | 100 | 26.59 | 100 | 93.66 | 88.09 | 88.06 | 85.27 | 94.78 | 63.13 | 56.5 | 51.34 | 44.81 | 48.96 |
| UD_Spanish-AnCora | 99.98 | 99.07 | 99.98 | 98.78 | 98.67 | 98.59 | 97.97 | 99.19 | 92.21 | 90.01 | 86.05 | 84.22 | 85.2 |
| UD_Spanish-GSD | 99.96 | 95.97 | 99.87 | 96.69 | 99.87 | 96.4 | 94.44 | 98.44 | 89.61 | 86.73 | 81.22 | 73.96 | 79.19 |
| UD_Swedish_Sign_Language-SSLC | 100 | 5.88 | 100 | 74.82 | 76.6 | 100 | 72.34 | 100 | 23.76 | 11.7 | 11.81 | 9.45 | 11.81 |
| UD_Swedish-LinES | 99.94 | 86.99 | 99.94 | 96.97 | 94.58 | 90.11 | 87.33 | 96.79 | 87.1 | 83.06 | 80.76 | 67.97 | 77.44 |
| UD_Swedish-Talbanken | 99.97 | 98.85 | 99.97 | 97.65 | 96.57 | 96.7 | 95.63 | 97.51 | 88.96 | 85.91 | 83.59 | 79.17 | 80.78 |
| UD_Tamil-TTB | 99.58 | 95.08 | 91.42 | 82.6 | 78.8 | 81.89 | 78.1 | 85.14 | 61.23 | 55.76 | 53.43 | 46.4 | 49.61 |
| UD_Telugu-MTG | 100 | 97.95 | 100 | 92.93 | 92.93 | 99.17 | 92.93 | 100 | 89.32 | 79.89 | 74.88 | 71.25 | 74.88 |
| UD_Turkish-IMST | 99.89 | 97.62 | 98.07 | 94.21 | 93.43 | 92.08 | 90.27 | 94.92 | 70.78 | 64.5 | 61.62 | 56.04 | 59.6 |
| UD_Ukrainian-IU | 99.81 | 96.65 | 99.79 | 96.77 | 92.49 | 92.53 | 91.31 | 96.49 | 87.11 | 83.86 | 80.51 | 73.38 | 77.28 |
| UD_Upper_Sorbian-UFAL | 89.72 | 79.45 | 89.72 | 58.57 | 89.72 | 38.64 | 36.39 | 52.64 | 34.25 | 23.61 | 17.3 | 4.18 | 8.62 |
| UD_Urdu-UDTB | 100 | 98.88 | 100 | 94.42 | 92.62 | 84.21 | 80.36 | 95.62 | 88.3 | 82.78 | 77.06 | 59.48 | 74.75 |
| UD_Uyghur-UDT | 99.79 | 86.9 | 99.79 | 89.45 | 91.92 | 87.92 | 80.54 | 96.16 | 75.55 | 63.61 | 57 | 46.06 | 54.39 |
| UD_Vietnamese-VTB | 87.25 | 93.15 | 87.25 | 79.5 | 77.9 | 87.02 | 77.87 | 87.2 | 53.63 | 48.16 | 44.88 | 42.17 | 44.85 |
| UD_Wolof-WTB | 99.97 | 91.06 | 99.42 | 94.09 | 94.03 | 93.11 | 91.26 | 94.6 | 83.25 | 77.05 | 70.94 | 64.25 | 66.99 |
{: .compact #conll18-results .datatable }

## System Performance on NER Corpora

In the table below you can find the performance of Stanza's pretrained NER models. All numbers reported are micro-averaged F1 scores. We used canonical train/dev/test splits for all datasets except for the WikiNER datasets, for which we used random splits.  The Ukrainian model and its score [was provided by a user](https://github.com/stanfordnlp/stanza/issues/319).

| Language | Corpus | # Types | F1 |
| :------- | :----- | :-------- | :---- |
| Afrikaans | NCHLT | 4 | 80.08 |
| Arabic | AQMAR | 4 | 74.3 |
| Bulgarian *New in 1.2.1* | BSNLP 2019 | 5 | 83.21 |
| Chinese | OntoNotes | 18 | 79.2 |
| Dutch | CoNLL02 | 4 | 89.2 |
| Dutch | WikiNER | 4 | 94.8 |
| English | CoNLL03 | 4 | 92.1 |
| English | OntoNotes | 18 | 88.8 |
| Finnish *New in 1.2.1* | Turku | 6 | 87.04 |
| French | WikiNER | 4 | 92.9 |
| German | CoNLL03 | 4 | 81.9 |
| German | GermEval2014 | 4 | 85.2 |
| Hungarian *New in 1.2.1* | Combined | 4 | - |
| Italian *New in 1.2.3* | FBK | 3 | 87.92 |
| Myanmar *New in 1.3.1* | UCSY | 7 | 95.86 |
| Russian | WikiNER | 4 | 92.9 |
| Spanish | CoNLL02 | 4 | 88.1 |
| Spanish | AnCora | 4 | 88.6 |
| Ukrainian | languk | 4 | 86.05 |
| Vietnamese *New in 1.2.1* | VLSP | 4 | 82.44 |

### Notes on NER Corpora

We have provided links to all NER datasets used to train the released models on our [available NER models page](available_models.md#available-ner-models). Here we provide notes on how to find several of these corpora:

- **Afrikaans**: The Afrikaans data is part of [the NCHLT corpus of South African languages](https://repo.sadilar.org/handle/20.500.12185/299).  Van Huyssteen, G.B., Puttkammer, M.J., Trollip, E.B., Liversage, J.C., Eiselen, R. 2016. [NCHLT Afrikaans Named Entity Annotated Corpus. 1.0](https://hdl.handle.net/20.500.12185/299).


- **Bulgarian**: The Bulgarian BSNLP 2019 data is available from [the shared task page](http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html). You can also find their [dataset description paper](https://www.aclweb.org/anthology/W19-3709/).

- **Finnish**: The Turku dataset used for Finnish NER training can be found on [the Turku NLP website](https://turkunlp.org/fin-ner.html), and they also provide [a Turku NER dataset description paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.567.pdf).

- **Hungarian**: The dataset used for training our Hungarian NER system is a combination of 3 separate datasets. Two of these datasets can be found from [this Szeged page](https://rgai.inf.u-szeged.hu/node/130), and the third can be found in [this NYTK-NerKor github repo](https://github.com/nytud/NYTK-NerKor). A dataset description paper can also be found [here](http://www.inf.u-szeged.hu/projectdirs/hlt/papers/lrec_ne-corpus.pdf).

- **Italian**: The Italian FBK dataset was licensed to us from [FBK](https://dh.fbk.eu/).

- **Myanmar**: The Myanmar dataset is by special request from [UCSY](https://arxiv.org/ftp/arxiv/papers/1903/1903.04739.pdf).

- **Vietnamese**: The Vietnamese VLSP dataset is available by [request from VLSP](https://vlsp.org.vn/vlsp2018/eval/ner).

