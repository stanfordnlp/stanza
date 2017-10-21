"""
A basic example showing how to build a document and run a pipeline on it.
"""

from stanfordnlp.data_structures import Document
from stanfordnlp.pipeline import Pipeline

# create the document
simple_document = Document("Joe Smith went to Hawaii .\nHe enjoyed his vacation .")

# create the pipeline
pipeline = Pipeline("tokenize,ssplit")

# run the pipeline on the document
pipeline.annotate(simple_document)

# print out the json of the annotated document
print(simple_document.to_json())
