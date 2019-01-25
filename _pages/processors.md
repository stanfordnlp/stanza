---
title: Processors 
keywords: processor
permalink: '/processors.html'
---

| Name | Annotator class name | Generated Annotation | Description |
| --- | --- | --- | --- | 
| tokenize | TokenizeProcessor | Segments a `Document` into `Sentence`s, each containing a list of `Token`s. This processor doesn't expand multi-word tokens (see the [MWT expander](/mwt.html)). | Tokenizes the text and performs sentence segmentation. |
| mwt | MWTProcessor | Expands multi-word tokens into multiple words when they are predicted by the tokenizer. | Expands [multi-word tokens (MWT)](https://universaldependencies.org/u/overview/tokenization.html) predicted by the [tokenizer](/tokenize.html). |
| lemma | LemmaProcessor | Perform [lemmatization](https://en.wikipedia.org/wiki/Lemmatisation) on a `Token` using the `Token.word` and `Token.pos` value. The result can be accessed in `Token.lemma`. | Generates the word lemmas for all tokens in the corpus. |
| pos | POSProcessor | UPOS, XPOS, and UFeats annotations accessible through `Token`'s properties `pos`, `xpos`, and `ufeats`. | Labels tokens with their [universal POS (UPOS) tags](https://universaldependencies.org/u/pos/), treebank-specific POS (XPOS) tags, and [universal morphological features (UFeats)](https://universaldependencies.org/u/feat/index.html). |
| depparse | DepparseProcessor | Determines the syntactic head of each word in a sentence and the dependency relation between the two words that are accessible through `Token`'s `governor` and `dependency_relation` attributes. | Provides an accurate syntactic dependency parser. |