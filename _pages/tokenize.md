---
layout: page
title: Tokenization & Sentence Segmentation
keywords: tokenize, TokenizeProcessor, tokenization, sentence segmentation
permalink: '/tokenize.html'
nav_order: 4
parent: Neural Pipeline
---

## Description

Tokenization and sentence segmentation in Stanza are jointly performed by the `TokenizeProcessor`. This processor splits the raw input text into tokens and sentences, so that downstream annotation can happen at the sentence level. This processor can be invoked by the name `tokenize`.

| Name | Annotator class name | Requirement | Generated Annotation | Description |
| --- | --- | --- | --- | --- |
| tokenize | TokenizeProcessor | - | Segments a [`Document`](data_objects.md#document) into [`Sentence`](data_objects.md#sentence)s, each containing a list of [`Token`](data_objects.md#token)s. This processor also predicts which tokens are multi-word tokens, but leaves expanding them to the [MWTProcessor](mwt.md). | Tokenizes the text and performs sentence segmentation. |

## Options

The following options are available to configure the `TokenizeProcessor` when instantiating the [`Pipeline`](pipeline.md#pipeline):

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| tokenize_batch_size | int | 32 | When annotating, this argument specifies the maximum number of paragraphs to process as a minibatch for efficient processing. <br>**Caveat**: the larger this number is, the more working memory is required (main RAM or GPU RAM, depending on the computating device). |
| tokenize_pretokenized | bool | False | Assume the text is tokenized by white space and sentence split by newline.  Do not run a model. |
| tokenize_no_ssplit | bool | False | Assume the sentences are split by two continuous newlines (`\n\n`). Only run tokenization and disable sentence segmentation. |

## Example Usage

The `TokenizeProcessor` is usually the first processor used in the pipeline. It performs tokenization and sentence segmentation at the same time. After this processor is run, the input document will become a list of [`Sentence`](data_objects.md#sentence)s. Each [`Sentence`](data_objects.md#sentence) contains a list of [`Token`](data_objects.md#token)s, which can be accessed with the property `tokens`.

### Tokenization and Sentence Segmentation

Here is a simple example of performing tokenization and sentence segmentation on a piece of plaintext:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize')
doc = nlp('This is a test sentence for stanza. This is another sentence.')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
```

This code will generate the following output, which shows that the text is segmented into two sentences, each containing a few tokens:

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

You can also use the tokenizer just for sentence segmentation. To access segmented sentences, simply use

```python
print([sentence.text for sentence in doc.sentences])
```

### Tokenization without Sentence Segmentation

Sometimes you might want to tokenize your text given existing sentences (e.g., in machine translation). You can perform tokenization without sentence segmentation, as long as the sentences are split by two continuous newlines (`\n\n`) in the raw text. Just set `tokenize_no_ssplit` as `True` to disable sentence segmentation. Here is an example:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
doc = nlp('This is a sentence.\n\nThis is a second. This is a third.')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
```

As you can see in the output below, only two [`Sentence`](data_object.md#sentence)s resulted from this processing, where the second contains all the tokens in the second and third sentences if we were to perform sentence segmentation.

```
====== Sentence 1 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: a
id: 4   text: sentence
id: 5   text: .
====== Sentence 2 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: a
id: 4   text: second
id: 5   text: .
id: 6   text: This
id: 7   text: is
id: 8   text: a
id: 9   text: third
id: 10  text: .
```

Contrast this to Stanza's output when `tokenize_no_ssplit` is set to `False` (its default value):

```
====== Sentence 1 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: a
id: 4   text: sentence
id: 5   text: .
====== Sentence 2 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: a
id: 4   text: second
id: 5   text: .
====== Sentence 3 tokens =======
id: 1   text: This
id: 2   text: is
id: 3   text: a
id: 4   text: third
id: 5   text: .
```

Note that sentence segmentation is performed here as is normally the case.

### Start with Pretokenized Text

In some cases, you might have already tokenized your text, and just want to use Stanza for downstream processing.
In these cases, you can feed in pretokenized (and sentence split) text to the pipeline, as newline (`\n`) separated sentences, where each sentence is space separated tokens. Just set `tokenize_pretokenized` as `True` to bypass the neural tokenizer.

The code below shows an example of bypassing the neural tokenizer:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True)
doc = nlp('This is token.ization done my way!\nSentence split, too!')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
```

An alternative to passing in a string is to pass in a list of lists of strings, representing a document with sentences, each sentence a list of tokens.

The equivalent of our example above would be:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True)
doc = nlp([['This', 'is', 'token.ization', 'done', 'my', 'way!'], ['Sentence', 'split,', 'too!']])
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
```

As you can see in the output below, no further tokenization or sentence segmentation is performed (note how punctuation are attached to the end of tokens as well as inside of tokens.

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

Contrast this with Stanza's output when `tokenize_pretokenized` is set to `False`, Stanza would perform tokenization and sentence segmentation as it sees fits.

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

Stanza will also accept a pretokenized `Document` for further processing with this flag:

```python
import stanza

nlp_tokenized = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True)
nlp_pos = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
doc = nlp_tokenized('This is token.ization done my way!\nSentence split, too!')
doc = nlp_pos(doc)
print("{:C}".format(doc))

for i, sentence in enumerate(doc.sentences):
    print(sentence)
```

This will output the following (results may vary based on POS model used):

```
# text = This is token.ization done my way!
# sent_id = 0
1       This    _       PRON    DT      Number=Sing|PronType=Dem        0       _       _       start_char=0|end_char=4
2       is      _       AUX     VBZ     Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin   1       _       _       start_char=5|end_char=7
3       token.ization   _       PROPN   NNP     Number=Sing     2       _       _       start_char=8|end_char=21
4       done    _       VERB    VBN     Tense=Past|VerbForm=Part        3       _       _       start_char=22|end_char=26
5       my      _       PRON    PRP$    Case=Gen|Number=Sing|Person=1|Poss=Yes|PronType=Prs     4       _       _       start_char=27|end_char=29
6       way!    _       NOUN    NN      Number=Sing     5       _       _       start_char=30|end_char=34

# text = Sentence split, too!
# sent_id = 1
1       Sentence        _       NOUN    NN      Number=Sing     0       _       _       start_char=35|end_char=43
2       split,  _       ADJ     JJ      Degree=Pos      1       _       _       start_char=44|end_char=50
3       too!    _       PUNCT   .       _       2       _       _       start_char=51|end_char=55
```

### Use spaCy for Fast Tokenization and Sentence Segmentation

{% include alerts.html %}
{{ note }}
{{ "You can only use spaCy to tokenize English text for now, since spaCy tokenizer does not handle multi-word token expansion for other languages." | markdownify }}
{{ end }}

While our neural pipeline can achieve significantly higher accuracy, rule-based tokenizer such as [`spaCy`](https://spacy.io) runs much faster when processing large-scale text. We provide an interface to use [`spaCy`](https://spacy.io) as the tokenizer for English by simply specifying in the `processors` option. Please make sure you have successfully downloaded and installed [`spaCy`](https://spacy.io) and English models following their [usage guide](https://spacy.io/usage).

To perform tokenization and sentence segmentation with [`spaCy`](https://spacy.io), simply set the package for the `TokenizeProcessor` to `spacy`, as in the following example:

```python
import stanza

nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'}) # spaCy tokenizer is currently only allowed in English pipeline.
doc = nlp('This is a test sentence for stanza. This is another sentence.')
for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
```

This will allow us to tokenize the text with Spacy and use it in downstream annotations in Stanza. The output is:

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

Most training-only options are documented in the [argument parser](https://github.com/stanfordnlp/stanza/blob/main/stanza/models/tokenizer.py#L12) of the tokenizer.

Note that to train the tokenizer for Vietnamese, one needs to postprocess the character labels generated from the plain text file and the CoNLL-U file to form syllable-level labels, which is automatically handled if you are using the training scripts we provide.
