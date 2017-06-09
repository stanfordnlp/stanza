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
  from corenlp import Pipeline

  # document.dat contains a serialized Document.
  with open('document.txt', 'r') as f:
    doc = f.read()

  pipeline = Pipeline(annotators='tokenize ssplit pos lemma ner'.split())
  ann = pipeline.annotate(doc)

  # You can access annotations using ann.
  sentence = ann.sentence[0]

  # You can access any property within a sentence.
  print(sentence.text)

  # Likewise for tokens
  token = sentence.token[0]
  print(token.lemma)

See `test_pipeline.py` for more examples.

Annotation Service Usage::
  from corenlp import AnnotatorBackend
  class NeuralNER(AnnotatorBackend):
    pass

See `test_service.py` for more examples.
