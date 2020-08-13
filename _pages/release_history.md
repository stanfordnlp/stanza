---
layout: page
title: Release History
keywords: history
permalink: '/release_history.html'
nav_order: 3
toc: false
parent: Resources
---

Note that prior to version 1.0.0, the Stanza library was named as "StanfordNLP". To install historical versions prior to to v1.0.0, you'll need to run `pip install stanfordnlp`.

| Version | Date&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Notes |
| :--- | :----------------------------------- | :--- |
| 1.1.1 | 2020-08-13 | This release features support for extending the capability of the Stanza pipeline with customized processors, a new sentiment analysis tool for English/German/Chinese, improvements to the CoreNLPClient functionality (including compatibility with CoreNLP 4.1.0), new models for a few languages (including Thai, which is supported for the first time in Stanza), new biomedical and clinical English packages, alternative servers for downloading resource files, and various improvements and bugfixes ([full release log](https://github.com/stanfordnlp/stanza/releases/tag/v1.1.1)). |
| 1.0.1 | 2020-04-27 | This is a maintenance release of Stanza. It features new support for jieba as Chinese tokenizer, faster lemmatizer implementation, improved compatibility with CoreNLP v4.0.0, and several bugfixes including correct character offsets in NER output and correct Vietnamese tokenization outputs ([full release log](https://github.com/stanfordnlp/stanza/releases/tag/v1.0.1)). |
| 1.0.0 | 2020-03-17 | This release introduces new multi-lingual named entity recognition (NER) support for 8 languages, expanded UD pipeline coverage of 66 languages, improved download and pipeline interfaces, improved document object interfaces, Anaconda installation support, improved neural lemmatizer, spaCy tokenization integration, and various other enhancements and bugfixes ([full release log](https://github.com/stanfordnlp/stanza/releases/tag/v1.0.0)). |
| 0.2.0 | 2019-05-16 | This release introduces substantially reduced model size, substantial lemmatizer speed up and more options for customizing server start up and requests ([full release log](https://github.com/stanfordnlp/stanza/releases/tag/v0.2.0)). |
| 0.1.2 | 2019-02-26 | This release introduces support for pretokenized text, speed ups in the POS/Feats tagger and various bug fixes ([full release log](https://github.com/stanfordnlp/stanza/releases/tag/v0.1.2)). |
| 0.1.0 | 2019-01-29 | Initial release of StanfordNLP ([full release log](https://github.com/stanfordnlp/stanza/releases/tag/v0.1.0)). |
