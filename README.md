# stanfordnlp
The Stanford NLP group's official Python code.  It contains packages for running our latest fully neural pipeline from the CoNLL 2018 Shared Task and for accessing the Java Stanford CoreNLP server.

### References

If you use the neural tokenizer, multi-word token expansion model, lemmatizer, POS/morphological features tagger, or dependency parser in your research, please kindly cite our CoNLL 2018 Shared Task [system description paper](http://universaldependencies.org/conll18/proceedings/pdf/K18-2016.pdf)

```bibtex
@InProceedings{qi2018universal,
  author    = {Qi, Peng  and  Dozat, Timothy  and  Zhang, Yuhao  and  Manning, Christopher D.},
  title     = {Universal Dependency Parsing from Scratch},
  booktitle = {Proceedings of the {CoNLL} 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
  month     = {October},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  pages     = {160--170},
  url       = {http://www.aclweb.org/anthology/K18-2016}
}
```
If you use the CoreNLP server, please cite the software package and the respective modules as described [here](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers) ("Citing Stanford CoreNLP in papers").

## Requirements

* bash 4.3.48 (the scripts for training and evaluating CoNLL 2018 models use new features in bash)

* python 3.6.5

* numpy 1.15.4
* protobuf 3.6.1
* requests 2.10.1
* torch 0.4.1 or above (only if you want to use the fully neural pipeline from the CoNLL 2018 Shared Task)

### Note about versions
The versions listed above reflect what the code has been tested with.  It is possible older versions can be used in some cases.

### Note about GPU's
The provided models only run on GPU's at this time.  It is possible to train models that run on a CPU.


## Setup

You can install the package as follows:

```
git clone git@github.com:stanfordnlp/stanfordnlp.git
cd stanfordnlp
pip install -e .
```

## Using the Fully Neural Pipeline

### Trained Models

We currently provide models for all of the treebanks in the CoNLL 2018 Shared Task.   You can find links to these models in the table below.

| language         | version    | zip file |
| :--------------- | :--------- | :------- |
| UD_English_EWT   | 1.0.0      | [download](http://nlp.stanford.edu/software/conll_2018/english_ewt_models.zip) |

### Pipeline

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

### Training your own models

The following models can be trained with this code

```
tokenizer
mwt_expander
lemmatizer
tagger
parser
```

#### Setup

Before training and evaluating, you need to set up the `scripts/config.sh`

Change `/path/to/CoNLL18` and `/path/to/word2vec` appropriately to where you have downloaded these resources.

#### Training

To train a model, run this command from the root directory:

```
bash scripts/run_${task}.sh ${treebank} ${gpu_num}
```

For example:

```
bash scripts/run_tokenize.sh UD_English-EWT 0
```

For the dependency parser, you also need to specify `gold|predicted` for the tag type in the training/dev data. 

```
bash scripts/run_depparse.sh UD_English-EWT 0 predicted
```

Models will be saved to the `saved_models` directory.

### Evaluation

Once you have trained all of the models for the pipeline, you can evaluate the full end-to-end system with this command:

```
bash scripts/run_ete.sh UD_English-EWT 0 test
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
