Stanford CoreNLP Python Interface
=================================

.. image:: https://travis-ci.org/stanfordnlp/python-stanford-corenlp.svg?branch=master
    :target: https://travis-ci.org/stanfordnlp/python-stanford-corenlp

This package contains a python interface for `Stanford CoreNLP
<https://github.com/stanfordnlp/CoreNLP>`_ that contains a reference
implementation to interface with the `Stanford CoreNLP server
<https://stanfordnlp.github.io/CoreNLP/corenlp-server.html>`_.
The package also contains a base class to expose a python-based annotation
provider (e.g. your favorite neural NER system) to the CoreNLP
pipeline via a lightweight service.

To use the package, first download the `official java CoreNLP release 
<https://stanfordnlp.github.io/CoreNLP/#download>`_, unzip it, and define an environment
variable :code:`$CORENLP_HOME` that points to the unzipped directory.

You can also install this package from `PyPI <https://pypi.python.org/pypi/stanford-corenlp/>`_ using :code:`pip install stanford-corenlp` 

----

Command Line Usage
------------------
Probably the easiest way to use this package is through the `annotate` command-line utility::

    usage: annotate [-h] [-i INPUT] [-o OUTPUT] [-f {json}]
                    [-a ANNOTATORS [ANNOTATORS ...]] [-s] [-v] [-m MEMORY]
                    [-p PROPS [PROPS ...]]

    Annotate data

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Input file to process; each line contains one document
                            (default: stdin)
      -o OUTPUT, --output OUTPUT
                            File to write annotations to (default: stdout)
      -f {json}, --format {json}
                            Output format
      -a ANNOTATORS [ANNOTATORS ...], --annotators ANNOTATORS [ANNOTATORS ...]
                            A list of annotators
      -s, --sentence-mode   Assume each line of input is a sentence.
      -v, --verbose-server  Server is made verbose
      -m MEMORY, --memory MEMORY
                            Memory to use for the server
      -p PROPS [PROPS ...], --props PROPS [PROPS ...]
                            Properties as a list of key=value pairs


We recommend using `annotate` in conjuction with the wonderful `jq`
command to process the output. As an example, given a file with a
sentence on each line, the following command produces an equivalent
space-separated tokens::

    cat file.txt | annotate -s -a tokenize | jq '[.tokens[].originalText]' > tokenized.txt


Annotation Server Usage
-----------------------

.. code-block:: python

  import corenlp

  text = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP."

  # We assume that you've downloaded Stanford CoreNLP and defined an environment
  # variable $CORENLP_HOME that points to the unzipped directory.
  # The code below will launch StanfordCoreNLPServer in the background
  # and communicate with the server to annotate the sentence.
  with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
    ann = client.annotate(text)

  # You can access annotations using ann.
  sentence = ann.sentence[0]

  # The corenlp.to_text function is a helper function that
  # reconstructs a sentence from tokens.
  assert corenlp.to_text(sentence) == text

  # You can access any property within a sentence.
  print(sentence.text)

  # Likewise for tokens
  token = sentence.token[0]
  print(token.lemma)

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

See `test_client.py` and `test_protobuf.py` for more examples. Props to
@dan-zheng for tokensregex/semgrex support.


Annotation Service Usage
------------------------

*NOTE*: The annotation service allows users to provide a custom
annotator to be used by the CoreNLP pipeline. Unfortunately, it relies
on experimental code internal to the Stanford CoreNLP project is not yet
available for public use.

.. code-block:: python

    import corenlp
    from .happyfuntokenizer import Tokenizer

    class HappyFunTokenizer(Tokenizer, corenlp.Annotator):
        def __init__(self, preserve_case=False):
            Tokenizer.__init__(self, preserve_case)
            corenlp.Annotator.__init__(self)

        @property
        def name(self):
            """
            Name of the annotator (used by CoreNLP)
            """
            return "happyfun"

        @property
        def requires(self):
            """
            Requires has to specify all the annotations required before we
            are called.
            """
            return []

        @property
        def provides(self):
            """
            The set of annotations guaranteed to be provided when we are done.
            NOTE: that these annotations are either fully qualified Java
            class names or refer to nested classes of
            edu.stanford.nlp.ling.CoreAnnotations (as is the case below).
            """
            return ["TextAnnotation",
                    "TokensAnnotation",
                    "TokenBeginAnnotation",
                    "TokenEndAnnotation",
                    "CharacterOffsetBeginAnnotation",
                    "CharacterOffsetEndAnnotation",
                   ]

        def annotate(self, ann):
            """
            @ann: is a protobuf annotation object.
            Actually populate @ann with tokens.
            """
            buf, beg_idx, end_idx = ann.text.lower(), 0, 0
            for i, word in enumerate(self.tokenize(ann.text)):
                token = ann.sentencelessToken.add()
                # These are the bare minimum required for the TokenAnnotation
                token.word = word
                token.tokenBeginIndex = i
                token.tokenEndIndex = i+1

                # Seek into the txt until you can find this word.
                try:
                    # Try to update beginning index
                    beg_idx = buf.index(word, beg_idx)
                except ValueError:
                    # Give up -- this will be something random
                    end_idx = beg_idx + len(word)

                token.beginChar = beg_idx
                token.endChar = end_idx

                beg_idx, end_idx = end_idx, end_idx

    annotator = HappyFunTokenizer()
    # Calling .start() will launch the annotator as a service running on
    # port 8432 by default.
    annotator.start()

    # annotator.properties contains all the right properties for
    # Stanford CoreNLP to use this annotator. 
    with corenlp.CoreNLPClient(properties=annotator.properties, annotators="happyfun ssplit pos".split()) as client:
        ann = client.annotate("RT @ #happyfuncoding: this is a typical Twitter tweet :-)")

        tokens = [t.word for t in ann.sentence[0].token]
        print(tokens)


See `test_annotator.py` for more examples.
