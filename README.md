<p align="center"><img src="images/stanza-logo.png" height="100px"/></p>

<h2 align="center">
    <p>Stanza: A Python NLP Library for Many Human Languages</p>
</h2>

<p align="center">
    <a href="https://travis-ci.com/stanfordnlp/stanfordnlp">
        <img alt="Travis Status" src="https://travis-ci.com/stanfordnlp/stanza.svg?token=RPNzRzNDQRoq2x3J2juj&branch=master">
    </a>
    <a href="https://pypi.org/project/stanfordnlp/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/stanza.svg?colorB=blue">
    </a>
    <img alt="Conda Versions" src="https://img.shields.io/conda/vn/yuhaozhang/stanza?color=blue&label=conda">
    <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/stanza.svg?colorB=blue">
</p>

The Stanford NLP Group's official Python NLP library. It contains support for running various accurate natural language processing tools on 60+ languages and for accessing the Java Stanford CoreNLP software from Python. For detailed information please visit our [official website](https://stanfordnlp.github.io/stanza/).

### References

If you use our neural pipeline including the tokenizer, the multi-word token expansion model, the lemmatizer, the POS/morphological features tagger, the dependency parser, or the named entity recognition tool in your research, please kindly cite our CoNLL 2018 Shared Task [system description paper](https://nlp.stanford.edu/pubs/qi2018universal.pdf):

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
The PyTorch implementation of the neural pipeline in this repository is due to [Peng Qi](http://qipeng.me), [Yuhao Zhang](http://yuhao.im), and [Yuhui Zhang](https://cs.stanford.edu/~yuhuiz/), with help from [Jason Bolton](mailto:jebolton@stanford.edu) and [Tim Dozat](https://web.stanford.edu/~tdozat/).

If you use the CoreNLP software through Stanza, please cite the CoreNLP software package and the respective modules as described [here](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers) ("Citing Stanford CoreNLP in papers"). The CoreNLP client is mostly written by [Arun Chaganty](http://arun.chagantys.org/), and [Jason Bolton](mailto:jebolton@stanford.edu) spearheaded merging the two projects together.

## Issues and Usage Q&A

To ask questions, report issues or request features, please use the [GitHub Issue Tracker](https://github.com/stanfordnlp/stanza/issues).

## Setup

Stanza supports Python 3.6 or later. We strongly recommend that you install Stanza from PyPI. If you already have [pip](https://pip.pypa.io/en/stable/installing/), the Python package manager, installed on your system, simply run:
```bash
pip install stanza
```
this should also help resolve all of the dependencies of Stanza, for instance [PyTorch](https://pytorch.org/) 1.0.0 or above.

If you currently have a previous version of `stanza` installed, use:
```bash
pip install stanza -U
```

Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of Stanza and training your own models. For this option, run
```bash
git clone https://github.com/stanfordnlp/stanza.git
cd stanza
pip install -e .
```

## Running Stanza

### Getting Started with the neural pipeline

To run your first Stanza pipeline, simply following these steps in your Python interactive interpreter:

```python
>>> import stanza
>>> stanza.download('en')   # This downloads the English models for the neural pipeline
# IMPORTANT: The above line prompts you before downloading, which doesn't work well in a Jupyter notebook.
# To avoid a prompt when using notebooks, instead use: >>> stanza.download('en', force=True)
>>> nlp = stanza.Pipeline() # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

The last command will print out the words in the first sentence in the input string (or [`Document`](https://stanfordnlp.github.io/stanza/data_objects.html#document), as it is represented in Stanza), as well as the indices for the word that governs it in the Universal Dependencies parse of that sentence (its "head"), along with the dependency relation between the words. The output should look like:

```
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')
```

See [our getting started guide](https://stanfordnlp.github.io/stanza/installation_usage.html#getting-started) for more details.

### Accessing Java Stanford CoreNLP software

Aside from the neural pipeline, this package also includes an official wrapper for acessing the Java Stanford CoreNLP software with Python code.

There are a few initial setup steps.

* Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and models for the language you wish to use
* Put the model jars in the distribution folder
* Tell the Python code where Stanford CoreNLP is located by setting the `CORENLP_HOME` environment variable (e.g., in *nix): `export CORENLP_HOME=/path/to/stanford-corenlp-full-2018-10-05`

We provide [comprehensive examples](https://stanfordnlp.github.io/stanza/corenlp_client.html) in our documentation that show how one can use CoreNLP through Stanza and extract various annotations from it.

### Online Colab Notebooks

To get your started, we also provide interactive Jupyter notebooks in the `demo` folder. You can also open these notebooks and run them interactively on [Google Colab](https://colab.research.google.com). To view all available notebooks, follow these steps:

* Go to the [Google Colab website](https://colab.research.google.com)
* Navigate to `File` -> `Open notebook`, and choose `GitHub` in the pop-up menu
* Note that you do **not** need to give Colab access permission to your github account
* Type `stanfordnlp/stanza` in the search bar, and click enter

### Trained Models for the Neural Pipeline

We currently provide models for all of the [Universal Dependencies](https://universaldependencies.org/) treebanks v2.5, as well as NER models for a few widely-spoken languages. You can find instructions for downloading and using these models [here](https://stanfordnlp.github.io/stanza/models.html).

### Batching To Maximize Pipeline Speed

To maximize speed performance, it is essential to run the pipeline on batches of documents. Running a for loop on one sentence at a time will be very slow. The best approach at this time is to concatenate documents together, with each document separated by a blank line (i.e., two line breaks `\n\n`).  The tokenizer will recognize blank lines as sentence breaks. We are actively working on improving multi-document processing.

## Training your own neural pipelines

All neural modules in this library can be trained with your own data. The tokenizer, the multi-word token (MWT) expander, the POS/morphological features tagger, the lemmatizer and the dependency parser require [CoNLL-U](https://universaldependencies.org/format.html) formatted data, while the NER model requires the BIOES format. Currently, we do not support model training via the `Pipeline` interface. Therefore, to train your own models, you need to clone this git repository and set up from source.

For detailed step-by-step guidance on how to train and evaluate your own models, please visit our [training documentation](https://stanfordnlp.github.io/stanza/training.html).

## LICENSE

Stanza is released under the Apache License, Version 2.0. See the [LICENSE](https://github.com/stanfordnlp/stanza/blob/master/LICENSE) file for more details.
