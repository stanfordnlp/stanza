---
title: Pipeline
keywords: pipeline
permalink: '/pipeline.html'
---

## Pipeline

Users of StanfordNLP can process documents by building a [`Pipeline`](pipeline.md) with the desired `Processor` units.  The pipeline takes in a [`Document`](data_objects.md#document)
object or raw text, runs the processors in succession, and returns an annotated [`Document`](data_objects.md#document).

## Options

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| lang | str | "en" | Use recommended models for this language. |
| dir | str | ~/stanfordnlp_resources | Directory for storing the models. |
| package | str | "default" | Package to use. |
| processors | dict or str | {} | Processors to use. Support comma-seperated string or dictionary. |
| logging_level | str | None | Control infomation to print. |
| verbose | str | True | Simplified for logging_level. |
| use_gpu | bool | True | Attempt to use a GPU if possible. |
| kwargs | - | - | Other arguments for processors. |

Options for each of the individual processors can be specified when building the pipeline.  See the individual processor pages for descriptions.

## Usage

### Basic Example

```python
import stanfordnlp

stanfordnlp.download("en") # Download the default English models
nlp = stanfordnlp.Pipeline("en", processors='tokenize,pos', use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
doc = nlp("Barack Obama was born in Hawaii.") # Run the pipeline on input text
print(doc) # Look at the result
```

### Specifying A Full Config 

```python
import stanfordnlp

config = {
	'processors': 'tokenize,mwt,pos,lemma,depparse', # Comma-separated list of processors to use
	'lang': 'fr', # Language code for the language to build the Pipeline in
	'tokenize_model_path': './fr_gsd_models/fr_gsd_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
	'mwt_model_path': './fr_gsd_models/fr_gsd_mwt_expander.pt',
	'pos_model_path': './fr_gsd_models/fr_gsd_tagger.pt',
	'pos_pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt',
	'lemma_model_path': './fr_gsd_models/fr_gsd_lemmatizer.pt',
	'depparse_model_path': './fr_gsd_models/fr_gsd_parser.pt',
	'depparse_pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt'
}
nlp = stanfordnlp.Pipeline(**config) # Initialize the pipeline using a configuration dict
doc = nlp("Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie.") # Run the pipeline on input text
print(doc) # Look at the result
```

### Accessing Word Information

After a pipeline is run, a `Document` object will be created and populated with annotation data.
A `Document` contains a list of `Sentences`s, and a `Sentence` contains a list of `Token`s and
`Word`s. For the most part `Token`s and `Word`s overlap, but some tokens can be divided into
mutiple words, for instance the French token `aux` is divided into the words `à` and `les`.  The
dependency parses are derived over words.

In this code example, the `Document` is named `doc`.  After the text is annotated, the for loops
go through each sentence `sent` in `doc.sentences` and each word `word` in `sent.words` and information
about the word is printed out, specifically `word.text`, `word.lemma`, `word.upos`, and `word.xpos`.

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline("en")
doc = nlp("Barack Obama was born in Hawaii.")
print(*[f'text: {word.text+" "}\tlemma: {word.lemma}\tupos: {word.upos}\txpos: {word.xpos}' for sent in doc.sentences for word in sent.words], sep='\n')

```

The following output is generated:

```
text: Barack 	lemma: Barack	upos: PROPN	xpos: NNP
text: Obama 	lemma: Obama	upos: PROPN	xpos: NNP
text: was 	lemma: be	upos: AUX	xpos: VBD
text: born 	lemma: bear	upos: VERB	xpos: VBN
text: in 	lemma: in	upos: ADP	xpos: IN
text: Hawaii 	lemma: Hawaii	upos: PROPN	xpos: NNP
text: . 	lemma: .	upos: PUNCT	xpos: .
```


### Running On Pre-Tokenized Text

If you set the `tokenize_pretokenized` option, the text will be interpreted as already tokenized on white space and sentence split by newlines.
The tokenizer model will not be run.

```python
import stanfordnlp

config = {
        'processors': 'tokenize,pos',
        'tokenize_pretokenized': True,
        'pos_model_path': './en_ewt_models/en_ewt_tagger.pt',
        'pos_pretrain_path': './en_ewt_models/en_ewt.pretrain.pt',
        'pos_batch_size': 1000
         }
nlp = stanfordnlp.Pipeline(**config)
doc = nlp('Joe Smith lives in California .\nHe loves pizza .')
print(doc)
```

You can also provide a list of lists representing sentences and tokens.  Make sure to still set the `tokenize_pretokenized` option to `True`.
Each list will represent the tokens of a sentence.

```python
pretokenized_text = [['hello', 'world'], ['hello', 'world', 'again']]
doc = nlp(pretokenized_text)
print(doc)
```
