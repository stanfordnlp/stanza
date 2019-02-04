# StanfordNLP: A Python NLP Library for Many Human Languages

[![Travis Status](https://travis-ci.com/stanfordnlp/stanfordnlp.svg?token=RPNzRzNDQRoq2x3J2juj&branch=master)](https://travis-ci.com/stanfordnlp/stanfordnlp)
[![PyPI version](https://img.shields.io/pypi/v/stanfordnlp.svg?colorB=blue)](https://pypi.org/project/stanfordnlp/)

The Stanford NLP Group's official Python NLP library. It contains packages for running our latest fully neural pipeline from the CoNLL 2018 Shared Task and for accessing the Java Stanford CoreNLP server. For detailed information please visit our [official website](https://stanfordnlp.github.io/stanfordnlp/).

### References

If you use our neural pipeline including the tokenizer, the multi-word token expansion model, the lemmatizer, the POS/morphological features tagger, or the dependency parser in your research, please kindly cite our CoNLL 2018 Shared Task [system description paper](https://nlp.stanford.edu/pubs/qi2018universal.pdf):

```bibtex
@inproceedings{qi2018universal,
 address = {Brussels, Belgium},
 author = {Qi, Peng  and  Dozat, Timothy  and  Zhang, Yuhao  and  Manning, Christopher D.},
 booktitle = {Proceedings of the {CoNLL} 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
 month = {October},
 pages = {160--170},
 publisher = {Association for Computational Linguistics},
 title = {Universal Dependency Parsing from Scratch},
 url = {https://nlp.stanford.edu/pubs/qi2018universal.pdf},
 year = {2018}
}
```
The PyTorch implementation of the neural pipeline in this repository is due to [Peng Qi](http://qipeng.me) and [Yuhao Zhang](http://yuhao.im), with help from [Tim Dozat](https://web.stanford.edu/~tdozat/) and [Jason Bolton](mailto:jebolton@stanford.edu).

This release is not the same as Stanford's CoNLL 2018 Shared Task system. The tokenizer, lemmatizer, morphological features, and multi-word term systems are a cleaned up version of the shared task code, but in the competition we used a  [Tensorflow version](https://github.com/tdozat/Parser-v3) of the tagger and parser by [Tim Dozat](https://web.stanford.edu/~tdozat/), which has been approximately reproduced in PyTorch (though with a few deviations from the original) for this release.

If you use the CoreNLP server, please cite the CoreNLP software package and the respective modules as described [here](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers) ("Citing Stanford CoreNLP in papers"). The CoreNLP client is mostly written by [Arun Chaganty](http://arun.chagantys.org/), and [Jason Bolton](mailto:jebolton@stanford.edu) spearheaded merging the two projects together.

## Issues and Usage Q&A

Please use the following channels for questions and issue reports.

| Purpose | Channel |
|---|---|
| Usage Q&A | [Google Group](https://groups.google.com/forum/#!forum/stanfordnlp) |
| Bug Reports and Feature Requests | [GitHub Issue Tracker](https://github.com/stanfordnlp/stanfordnlp/issues) |

## Setup

StanfordNLP supports Python 3.6 or later. We strongly recommend that you install StanfordNLP from PyPI. If you already have [pip installed](https://pip.pypa.io/en/stable/installing/), simply run
```bash
pip install stanfordnlp
```
this should also help resolve all of the dependencies of StanfordNLP, for instance [PyTorch](https://pytorch.org/) 1.0.0 or above.

Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of StanfordNLP and training your own models. For this option, run
```bash
git clone git@github.com:stanfordnlp/stanfordnlp.git
cd stanfordnlp
pip install -e .
```

## Running StanfordNLP

### Getting Started with the neural pipeline

To run your first StanfordNLP pipeline, simply following these steps in your Python interactive interpreter:

```python
>>> import stanfordnlp
>>> stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
>>> nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

The last command will print out the words in the first sentence in the input string (or `Document`, as it is represented in StanfordNLP), as well as the indices for the word that governs it in the Universal Dependencies parse of that sentence (its "head"), along with the dependency relation between the words. The output should look like:

```
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')
```

**Note:** If you are running into issues like `OSError: [Errno 22] Invalid argument`, it's very likely that you are affected by a [known Python issue](https://bugs.python.org/issue24658), and we would recommend Python 3.6.8 or later and Python 3.7.2 or later.

We also provide a multilingual [demo script](https://github.com/stanfordnlp/stanfordnlp/blob/master/demo/pipeline_demo.py) that demonstrates how one uses StanfordNLP in other languages than English, for example Chinese (traditional)

```bash
python demo/pipeline_demo.py -l zh
```

See [our getting started guide](https://stanfordnlp.github.io/stanfordnlp/installation_download.html#getting-started) for more details.

### Access to Java Stanford CoreNLP Server

Aside from the neural pipeline, this project also includes an official wrapper for acessing the Java Stanford CoreNLP Server with Python code.

There are a few initial setup steps.

* Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and models for the language you wish to use.
* Put the model jars in the distribution folder
* Tell the python code where Stanford CoreNLP is located: `export CORENLP_HOME=/path/to/stanford-corenlp-full-2018-10-05`

We provide another [demo script](https://github.com/stanfordnlp/stanfordnlp/blob/master/demo/corenlp.py) that shows how one can use the CoreNLP client and extract various annotations from it.


### Trained Models for the Neural Pipeline

We currently provide models for all of the treebanks in the CoNLL 2018 Shared Task. You can find instructions for downloading and using these models [here](https://stanfordnlp.github.io/stanfordnlp/installation_download.html#models-for-human-languages).

### Batching To Maximize Pipeline Speed

To maximize speed performance, it is essential to run the pipeline on batches of documents. Running a for loop
on one sentence at a time will be very slow. The best approach at this time is to concatenate documents together,
with each document separated by a blank line (i.e., two line breaks `\n\n`).  The tokenizer will recognize blank lines as sentence breaks.
We are actively working on improving multi-document processing.

## Training your own neural pipelines

All neural modules in this library, including the tokenizer, the multi-word token (MWT) expander, the POS/morphological features tagger, the lemmatizer and the dependency parser, can be trained with your own [CoNLL-U](https://universaldependencies.org/format.html) format data. Currently, we do not support model training via the `Pipeline` interface. Therefore, to train your own models, you need to clone this git repository and set up from source.

For detailed step-by-step guidance on how to train and evaluate your own models, please visit our [training documentation](https://stanfordnlp.github.io/stanfordnlp/training.html).

## LICENSE

StanfordNLP is released under the Apache License, Version 2.0. See the [LICENSE](https://github.com/stanfordnlp/stanfordnlp/blob/master/LICENSE) file for more details.
