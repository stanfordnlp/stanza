---
title: TokenizeProcessor
keywords: tokenize
permalink: '/tokenize.html'
---

## Description

Tokenizes the text and performs sentence segmentation.

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- | 
| tokenize | TokenizeProcessor | - | Segments a [`Document`](data_objects.md#document) into [`Sentence`](data_objects.md#sentence)s, each containing a list of [`Token`](data_objects.md#token)s. This processor also predicts which tokens are multi-word tokens, but leaves expanding them to the [MWTProcessor](mwt.md). | Tokenizes the text and performs sentence segmentation. |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| tokenize_batch_size | int | 32 | When annotating, this argument specifies the maximum number of paragraphs to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |
| tokenize_pretokenized | bool | False | Assume the text is tokenized by white space and sentence split by newline.  Do not run a model. |

## Example Usage

The `TokenizeProcessor` is usually the first processor used in the pipeline. It performs tokenization and sentence segmentation at the same time. After this processor is run, the input document will become a list of [`Sentence`](data_objects.md#sentence)s. The list of [`Token`](data_objects.md#token)s for [`Sentence`](data_objects.md#sentence) can then be accessed with the property `tokens`. 

### Tokenization and Sentence Segmentation

The code below shows an example of tokenization and sentence segmentation:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize')
doc = nlp('This is a test sentence for stanza. This is another sentence.')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
```

This code will generate the following output:

```
====== Sentence 1 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: a
id: 4   text: test
id: 5   text: sentence
id: 6   text: for
id: 7   text: stanza
id: 8   text: .
====== Sentence 2 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: another
id: 4   text: sentence
id: 5   text: .
```

### Start with Pretokenized Text

You can feed in pretokenized (and sentence split) text to the pipeline, as newline (`\n`) separated sentences, where each sentence is space separated tokens. Just set `tokenize_pretokenized` as `True` to bypass the neural tokenizer. 

The code below shows an example of bypassing the neural tokenizer:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True)
doc = nlp('This is token.ization done my way!\nSentence split, too!')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
```

Alternatively to passing in a string, you can also pass in a list of lists of strings, representing a document with sentences, each sentence a list of tokens. 

The equivalent of our example above would be:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True)
doc = nlp([['This', 'is', 'token.ization', 'done', 'my', 'way!'], ['Sentence', 'split,', 'too!']])
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
```

These codes will generate the following output:

```
====== Sentence 1 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: token.ization
id: 4   text: done
id: 5   text: my
id: 6   text: way!
====== Sentence 2 tokens =======
id: 1   text: Sentence
id: 2   text: split,
id: 3   text: too!
```

As can be seen from the output, tokenization and sentence split decisions are preserved. If `tokenize_pretokenized` were set to `False` and the input is a string, Stanza would have generated the following output with its own tokenization and sentence split:

```
====== Sentence 1 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: token
id: 4   text: .
id: 5   text: ization
id: 6   text: done
id: 7   text: my
id: 8   text: way
id: 9   text: !
====== Sentence 2 tokens =======
id: 1   text: Sentence
id: 2   text: split
id: 3   text: ,
id: 4   text: too
id: 5   text: !
```

### Use spaCy for Fast Tokenization and Sentence Segmentation

While our neural pipeline can achieve significantly higher accuracy, rule-based tokenizer such as [`spaCy`](https://spacy.io) runs much faster when processing large-scale text. We provide an interface to use [`spaCy`](https://spacy.io) as the tokenizer for English by simply specifying in the `processors` option. Please make sure you have successfully downloaded and installed [`spaCy`](https://spacy.io) and English models following the [guide](https://spacy.io/usage).

The code below shows an example of tokenization and sentence segmentation with [`spaCy`](https://spacy.io):

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'}) # spaCy tokenizer is currently only allowed in English pipeline.
doc = nlp('This is a test sentence for stanza. This is another sentence.')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
```

This code will generate the following output:

```
====== Sentence 1 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: a
id: 4   text: test
id: 5   text: sentence
id: 6   text: for
id: 7   text: stanza
id: 8   text: .
====== Sentence 2 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: another
id: 4   text: sentence
id: 5   text: .
```

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/master/stanza/models/tokenizer.py#L12) of the tokenizer.

Note that to train the tokenizer for Vietnamese, one would need to postprocess the character labels generated from the plain text file and the CoNLL-U file to form syllable-level labels, which is automatically handled if you are using the training scripts we provide.
