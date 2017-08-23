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

----

Annotation Server Usage
-----------------------

.. code-block:: python

  from corenlp import CoreNLPClient

  text = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP."

  # We assume that you've defined a variable $JAVANLP_HOME
  # that points to a Stanford CoreNLP checkout.
  # The code below will launch StanfordCoreNLPServer in the background
  # and communicate with the server to annotate the sentence.
  with CoreNLPClient(annotators="tokenize ssplit".split()) as client:
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

See `test_client.py` and `test_protobuf.py` for more examples.


Annotation Service Usage
------------------------

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
