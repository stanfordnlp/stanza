---
title: Data Objects 
keywords: data objects
permalink: '/data_objects.html'
---

This page describes the data objects used in StanfordNLP, and how they interact with each other.

## Document

A `Document` object holds the annotation of an entire document, and is automatically generated when a string is annotated by the `Pipeline`. It holds a collection of `Sentence`s, and can be seamlessly translated into a [CoNLL-U file](https://universaldependencies.org/format.html).

Objects of this class expose useful properties such as `text`, `sentences`, and `conll_file`.

## Sentence

A `Sentence` object represents a sentence (as is predicted by the [tokenizer](/tokenize.html)), and holds a list of the `Token`s in the sentence, as well as a list of all its `Word`s. It also processes the dependency parse as is predicted by the [parser](/depparse.html), through its member method `build_dependencies`.

Objects of this class expose useful properties such as `words`, `tokens`, and `dependencies`, as well as methods such as `print_tokens`, `print_words`, `print_dependencies`. 

## Token

A `Token` object holds a token, and a list of its underlying words. In the event that the token is a [multi-word token](https://universaldependencies.org/u/overview/tokenization.html) (e.g., French _au = à le_), the token will have a range `index` as described in the [CoNLL-U format specifications](https://universaldependencies.org/format.html#words-tokens-and-empty-nodes) (e.g., `3-4`), with its `word` property containing the underlying `Word`s. In other cases, the `Token` object will be a simple wrapper around one `Word` object, where its `words` property is a singleton.

Aside from `index` that gives the 1-based sentence index of the token and `words` that points to the underlying words, `Token` objects also provide `text` to access the raw form of the `Token`, which is a substring of the input text.

## Word

A `Word` object holds a syntactic word and all of its word-level annotations. In the example of multi-word tokens (MWT), these are generated as a result of [multi-word token expansion](/mwt.html), and are used in all downstream syntactic analyses such as tagging, lemmatization, and parsing. If a `Word` is the result from an MWT expansion, its `text` will usually not be found in the input raw text. Aside from multi-word tokens, `Word`s should be similar to the familiar "tokens" one would see elsewhere.

`Word` objects expose useful properties such as `index`, `text`, `lemma`, `pos` (which is an alias for `xpos`, the treebank-specific part-of-speech, e.g., `NN`), `upos` ([universal part-of-speech](https://universaldependencies.org/u/pos/), e.g., 'NOUN'), `feats` (morphological features), `governor` (governor/head in the dependency parse), `dependency_relation` (dependency relation between this word and its head), and `parent_token` (the `Token` object that this `Word` is part of).