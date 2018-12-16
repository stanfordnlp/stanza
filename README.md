# stanfordnlp
The Stanford NLP group's official Python code.  It contains packages for running our latest pipeline from the CoNLL 2018 shared task and for accessing the Java Stanford CoreNLP server.

## Requirements

Requires Python 3.6 or greater.

* protobuf 3.6.1
* requests 2.20.1
* torch 0.4.1

## Setup

You can install the package as follows:

```
git clone git@github.com:stanfordnlp/stanfordnlp.git
cd stanfordnlp
pip install -e .
```

## Training And Evaluating Models

The following models can be trained with this code

```
tokenizer
mwt_expander
lemmatizer
tagger
parser
```

### tokenize and mwt

Training the tokenizer and multi-word-token expander currently requires some extra set up, so the easiest way is to use provided scripts.

First you need to set up the directory structure the scripts expect.

```
# make directories to save models to
mkdir -p /path/to/stanfordnlp/saved_models/tokenize
mkdir /path/to/stanfordnlp/saved_models/mwt

# set up directory for tokenize data
mkdir -p /path/to/stanfordnlp/data/tokenize

# set up directory for mwt data
mkdir /path/to/stanfordnlp/data/mwt

# set up path to external data
mkdir /path/to/stanfordnlp/extern_data

# add CoNLL18 data to external data
ln -s /path/to/CoNLL18 /path/to/stanfordnlp/extern_data/CoNLL18

# add word2vec data to external data
ln -s /path/to/word2vec /path/to/stanfordnlp/extern_data/word2vec
```

```
# model will be saved to /path/to/stanfordnlp/saved_models/tokenize
bash scripts/run_tokenize.sh UD_English-EWT 0
```

```
# model will be saved to /path/to/stanfordnlp/saved_models/mwt
bash scripts/run_mwt.sh UD_French-GSD 0
```

### lemma, pos, and depparse

These three modules can be trained with the same python command.  One just needs to change the module name.

Here is an example of training the English-EWT dependency parser

```
# will save a model to /path/to/saved_models/depparse/en_ewt_parser.pt
# will save required word embeddings to /path/to/saved_models/depparse/en_ewt_parser.pretrain.pt
python -m stanfordnlp.models.parser --train_file train_data.conllu --eval_file dev_data.conllu --gold_file dev_data.conllu --output_file dev_data.predictions.conllu --lang en_ewt --shorthand en_ewt --mode train --save_dir /path/to/saved_models/depparse
```

A similar python command will enable evaluation on CoNLL-U files

```
# assumes appropriate resources in /path/to/saved_models/depparse
python -m stanfordnlp.models.parser --eval_file dev_data.conllu --gold_file dev_data.conllu --output_file dev_data.predictions.conllu --lang en_ewt --shorthand en_ewt --mode predict --save_dir /path/to/saved_models/depparse
```

## Trained Models

We currently provide models for all of the treebanks in the CoNLL 2018 Shared Task.   You can find links to these models in the table below.

| language         | version    | .tgz file |
| :--------------- | :--------- | :------- |
| UD_English_EWT   | 1.0.0      | [download](http://nlp.stanford.edu/software/conll_2018/english_ewt.models) |

## Pipeline

Once you have trained models, you can run a full NLP pipeline natively in Python, similar to running a pipeline with Stanford CoreNLP in Java.

The following demo code demonstrates how to run a pipeline

```
from stanfordnlp.pipeline import Document, Pipeline

# example documents
english_doc = Document('Barack Obama was born in Hawaii.  He was elected president in 2008.')

# example configs
english_config = {
    'processors': 'tokenize,lemma,pos,depparse',
    'tokenize.model_path': 'saved_models/tokenize/en_ewt_tokenizer.pt',
    'lemma.model_path': 'saved_models/lemma/en_ewt_lemmatizer.pt',
    'pos.pretrain_path': 'saved_models/pos/en_ewt_tagger.pretrain.pt',
    'pos.model_path': 'saved_models/pos/en_ewt_tagger.pt',
    'depparse.pretrain_path': 'saved_models/depparse/en_ewt_parser.pretrain.pt',
    'depparse.model_path': 'saved_models/depparse/en_ewt_parser.pt'
}

# build the pipeline
print('---')
print('loading pipeline...')
english_pipeline = Pipeline(config=english_config)
print('done loading pipeline')

# process the example document
print('---')
print('processing example document...')
english_pipeline.process(english_doc)
print('done')

# explore the processed document
print('---')
print('accessing NLP annotations...')
print('')
print('tokens of first sentence: ')
for tok in english_doc.sentences[0].tokens:
    print(tok.word + '\t' + tok.lemma + '\t' + tok.pos)
print('dependency parse of first sentence: ')
for dep_edge in english_doc.sentences[0].dependencies:
    print((dep_edge[0].word, dep_edge[1], dep_edge[2].word))
```

## Access to Java Stanford CoreNLP Server

This project also includes an official wrapper for acessing the Java Stanford CoreNLP Server with Python code.

### Setup 

There are  a few initial setup steps.

* Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and models for the language you wish to use.
* Put the model jars in the distribution folder
* Tell the python code where Stanford CoreNLP is located: `export CORENLP_HOME=/path/to/stanford-corenlp-full-2018-10-05`

### Demo

Here is some example Python code that will start a server, make an annotation request, and walk through the final annotation.

```
from stanfordnlp.server import CoreNLPClient

# example text
print('---')
print('input text')
print('')

text = "Chris Manning is a nice person.  He gives oranges to people."

print(text)

# set up the client
print('---')
print('starting up Java Stanford CoreNLP Server...')

# set up the client
client = CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner','depparse','coref'], memory='16G') 
    
# submit the request to the server
ann = client.annotate(text)
    
# get the first sentence
sentence = ann.sentence[0]
    
# get the dependency parse of the first sentence
print('---')
print('dependency parse of first sentence')
dependency_parse = sentence.basicDependencies
print(dependency_parse)
    
# get the first token of the first sentence
print('---')
print('first token of first sentence')
token = sentence.token[0]
print(token)
    
# get the part-of-speech tag
print('---')
print('part of speech tag of token')
token.pos
print(token.pos)
    
# get the named entity tag
print('---')
print('named entity tag of token')
print(token.ner)
    
# get an entity mention from the first sentence 
print('---')
print('first entity mention in sentence')
print(sentence.mentions[0])
    
# access the coref chain
print('---')
print('coref chains for the example')
print(ann.corefChain)
 
# Use tokensregex patterns to find who wrote a sentence.
pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
matches = client.tokensregex(text, pattern)
# sentences contains a list with matches for each sentence.
assert len(matches["sentences"]) == 1
# length tells you whether or not there are any matches in this
assert matches["sentences"][0]["length"] == 1
# You can access matches like most regex groups.
matches["sentences"][1]["0"]["text"] == "Chris wrote a simple sentence"
matches["sentences"][1]["0"]["1"]["text"] == "Chris"

# Use semgrex patterns to directly find who wrote what.
pattern = '{word:wrote} >nsubj {}=subject >dobj {}=object'
matches = client.semgrex(text, pattern)
# sentences contains a list with matches for each sentence.
assert len(matches["sentences"]) == 1
# length tells you whether or not there are any matches in this
assert matches["sentences"][0]["length"] == 1
# You can access matches like most regex groups.
matches["sentences"][1]["0"]["text"] == "wrote"
matches["sentences"][1]["0"]["$subject"]["text"] == "Chris"
matches["sentences"][1]["0"]["$object"]["text"] == "sentence"

```
