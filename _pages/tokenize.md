---
title: TokenizeProcessor
keywords: tokenize
permalink: '/tokenize.html'
---

## Description

Tokenizes the text and performs sentence segmentation.

| Property name | Processor class name | Generated Annotation |
| --- | --- | --- |
| tokenize | TokenizeProcessor | Segments a [`Document`](data_objects.md#document) into [`Sentence`](data_objects.md#sentence)s, each containing a list of [`Token`](data_objects.md#token)s. This processor also predicts which tokens are multi-word tokens, but leaves expanding them to the [MWT expander](mwt.md). |

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| tokenize_batch_size | int | 32 | When annotating, this argument specifies the maximum number of paragraphs to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |
| tokenize_pretokenized | bool | False | Assume the text is tokenized by white space and sentence split by newline.  Do not run a model. |

## Example Usage

The `tokenize` processor is usually the first processor used in the pipeline. It performs tokenization and sentence segmentation at the same time. After this processor is run, the input document will become a list of `Sentence`s. The list of tokens for sentence `sent` can then be accessed with `sent.tokens`. The code below shows an example of tokenization and sentence segmentation.

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')
doc = nlp("This is a test sentence for stanfordnlp. This is another sentence.")
for i, sentence in enumerate(doc.sentences):
    print(f"====== Sentence {i+1} tokens =======")
    print(*[f"index: {token.index.rjust(3)}\ttoken: {token.text}" for token in sentence.tokens], sep='\n')
```

This code will generate the following output:

```
====== Sentence 1 tokens =======
index:   1	token: This
index:   2	token: is
index:   3	token: a
index:   4	token: test
index:   5	token: sentence
index:   6	token: for
index:   7	token: stanfordnlp
index:   8	token: .
====== Sentence 2 tokens =======
index:   1	token: This
index:   2	token: is
index:   3	token: another
index:   4	token: sentence
index:   5	token: .
```

Alternatively, you can feed in pretokenized (and sentence split) text to the pipeline, as newline (`\n`) separated sentences, where each sentence is space separated tokens. Just set `tokenize_pretokenized` as `True` to bypass the neural tokenizer. The code below shows an example of bypassing the neural tokenizer.

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en', tokenize_pretokenized=True)
doc = nlp('This is token.ization done my way!\nSentence split, too!')
for i, sentence in enumerate(doc.sentences):
    print(f"====== Sentence {i+1} tokens =======")
    print(*[f"index: {token.index.rjust(3)}\ttoken: {token.text}" for token in sentence.tokens], sep='\n')
```

This code will generate the following output:

```
====== Sentence 1 tokens =======
index:   1	token: This
index:   2	token: is
index:   3	token: token.ization
index:   4	token: done
index:   5	token: my
index:   6	token: way!
====== Sentence 2 tokens =======
index:   1	token: Sentence
index:   2	token: split,
index:   3	token: too!
```

As can be seen from the output, tokenization and sentence split decisions are preserved. Alternatively to passing in a string, you can also pass in a list of lists of strings, representing a document with sentences, each sentence a list of tokens. The equivalent of our example above would be (note you don't have to set `tokenize_pretokenized` in this case):

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en', tokenize_pretokenized=True)
doc = nlp([['This', 'is', 'token.ization', 'done', 'my', 'way!'], ['Sentence', 'split,', 'too!']])
for i, sentence in enumerate(doc.sentences):
    print(f"====== Sentence {i+1} tokens =======")
    print(*[f"index: {token.index.rjust(3)}\ttoken: {token.text}" for token in sentence.tokens], sep='\n')
```

If `tokenize_pretokenized` were set to `False` and the input is a string, StanfordNLP would have generated the following output with its own tokenization and sentence split:

```
====== Sentence 1 tokens =======
index:   1	token: This
index:   2	token: is
index:   3	token: token
index:   4	token: .
index:   5	token: ization
index:   6	token: done
index:   7	token: my
index:   8	token: way
index:   9	token: !
====== Sentence 2 tokens =======
index:   1	token: Sentence
index:   2	token: split
index:   3	token: ,
index:   4	token: too
index:   5	token: !
```

## Training-Only Options

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanfordnlp/blob/master/stanfordnlp/models/tokenizer.py#L12) of the tokenizer.

Note that to train the tokenizer for Vietnamese, one would need to postprocess the character labels generated from the plain text file and the CoNLL-U file to form syllable-level labels, which is automatically handled if you are using the training scripts we provide.
