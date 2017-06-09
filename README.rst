Stanford CoreNLP Python Interface
=================================

This package contains a python interface for `Stanford CoreNLP
<https://github.com/stanfordnlp/CoreNLP>` that contains a reference
implementation to interface with the `Stanford CoreNLP server
<https://stanfordnlp.github.io/CoreNLP/corenlp-server.html>`. The
package also contains a base class to expose a python-based annotation
provider (e.g. your favorite neural NER system) to the CoreNLP
pipeline via a lightweight service.

----

Annotation Server Usage::
  from corenlp import CoreNLPClient

    text = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP."

    # We assume that you've defined a variable $JAVANLP_HOME
    # that points to a Stanford CoreNLP checkout.
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

See `test_client.py` and `test_protobuf.py` for more examples.
