# stanfordnlp
The Stanford NLP Group's official Python library.  It contains packages for running our latest fully neural pipeline from the CoNLL 2018 Shared Task and for accessing the Java Stanford CoreNLP server.

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
The PyTorch implementation of the neural pipeline in this repository is due to [Peng Qi](https://qipeng.me) and [Yuhao Zhang](https://yuhao.im), with help from [Tim Dozat](https://web.stanford.edu/~tdozat/), who is the main contributor to the [Tensorflow version](https://github.com/tdozat/Parser-v3) of the tagger and parser.

If you use the CoreNLP server, please cite the software package and the respective modules as described [here](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers) ("Citing Stanford CoreNLP in papers").

## Setup

StanfordNLP supports Python 3.6 and above. We strongly recommend that you install StanfordNLP from PyPI. If you already have [pip installed](https://pip.pypa.io/en/stable/installing/), simply run
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

To get run your first StanfordNLP pipeline, simply following these steps in your Python interactive interpreter:
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

We provide another [demo script](https://github.com/stanfordnlp/stanfordnlp/blob/master/demo/corenlp.py)


### Trained Models for the Neural Pipeline

We currently provide models for all of the treebanks in the CoNLL 2018 Shared Task. You can find instructions for downloading and using these models [here](https://stanfordnlp.github.io/stanfordnlp/installation_download.html#models-for-human-languages).

### Batching To Maximize Pipeline Speed

To maximize speed performance, it is essential to run the pipeline on batches of documents. Running a for loop
on one sentence at a time will be very slow. The best approach at this time is to concatenate documents together,
with each document separated by a blank line (i.e., two line breaks `\n\n`).  The tokenizer will recognize blank lines as sentence breaks.
We are actively working on improving multi-document processing.

## Training your own neural pipelines

The following models can be trained with this code: the tokenizer`, the multi-word token (MWT) expander, the lemmatizer, the POS/morphological features tagger, and the dependency parser.

To train your own models, you would need to clone the git repository and set up from source.

### Setup

Before training and evaluating, you need to set up the `scripts/config.sh`

Change `/path/to/CoNLL18` and `/path/to/word2vec` appropriately to where you have downloaded these resources.

For languages that had pretrained word2vec embeddings released from the CoNLL 2017 Shared Task, which can be found [here](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989/word-embeddings-conll17.tar?sequence=9&isAllowed=y). For the languages not in this list, we use the FastText embeddings from Facebook, which can be found [here](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). Once you download the embeddings, please make sure you arrange them in a similar fashion as the CoNLL 2017 archive, where you have a `<language_code>.xz` file under `/path/to/word2vec/<language_name>` for each language (language and writing standard in the case of Norwegian Bokmal and Norwegian Nynorsk).

### Training

To train a model, run this command from the root directory:

```bash
bash scripts/run_${task}.sh ${treebank} ${gpu_num}
```

For example:

```bash
bash scripts/run_tokenize.sh UD_English-EWT 0
```

For the dependency parser, you also need to specify `gold|predicted` for the tag type in the training/dev data.

```bash
bash scripts/run_depparse.sh UD_English-EWT 0 predicted
```

Models will be saved to the `saved_models` directory.

### Evaluation

Once you have trained all of the models for the pipeline, you can evaluate the full end-to-end system with this command:

```bash
bash scripts/run_ete.sh UD_English-EWT 0 test
```

