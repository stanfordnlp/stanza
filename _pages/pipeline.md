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
| models_dir | str | ~/stanfordnlp_resources | Directory for storing the models. |
| processors | str | "tokenize,<wbr>mwt,<wbr>pos,<wbr>lemma,<wbr>depparse" | List of processors to use. For a list of all processors supported, see [Processors Summary](processors.md). |
| treebank | str | None | Use models for this treebank. If not specified, `Pipeline` will look up the default treebank for the language requested. |
| use_gpu | bool | True | Attempt to use a GPU if possible. |

Options for each of the individual processors can be specified when building the pipeline.  See the individual processor pages for descriptions.

## Usage

### Basic Example

```python
import stanfordnlp

MODELS_DIR = '.'
stanfordnlp.download('en', MODELS_DIR) # Download the English models
nlp = stanfordnlp.Pipeline(processors='tokenize,pos', models_dir=MODELS_DIR, treebank='en_ewt', use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
doc = nlp("Barack Obama was born in Hawaii.") # Run the pipeline on input text
doc.sentences[0].print_tokens() # Look at the result
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
doc.sentences[0].print_tokens() # Look at the result
```

### Accessing Word Information

After a pipeline is run, a `Document` object will be created and populated with annotation data.  The document
contains a list of sentences `doc.sentences`, and a sentence contains a list of words `sent.words`.  You
can access various fields from the word, including `word.text`, `word.upos`, `word.xpos`, and `word.lemma`.  

```python
import stanfordnlp

nlp = stanfordnlp.Pipeline()
doc = nlp("Barack Obama was born in Hawaii.")
print(*[f'text: {word.text+" "}\tlemma: {word.lemma}\tupos: {word.upos}\txpos: {word.xpos}' for sent in doc.sentences for word in sent.words], sep='\n')

```

The code above will generate the following output:

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
print(doc.conll_file.conll_as_string())
```
