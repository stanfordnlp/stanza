---
layout: page
title: Client Basic Usage
keywords: CoreNLP, client, usage
permalink: '/client_usage.html'
nav_order: 2
parent: Stanford CoreNLP Client
# toc: false
---

After CoreNLP has been [properly set up](client_setup), you can start using the client functions to obtain CoreNLP annotations in Stanza.
Below are some basic examples of starting a server, making requests, and accessing various annotations from the returned Document object.
By default, CoreNLP Client uses `protobuf` for message passing. A full definition of our protocols (a.k.a., our supported annotations) can be found [here](https://github.com/stanfordnlp/stanza/blob/master/doc/CoreNLP.proto).

Apart from the following example code, we have also prepared an [interactive Jupyter notebook tutorial](https://github.com/stanfordnlp/stanza/blob/master/demo/StanfordNLP_CoreNLP_Interface.ipynb) to get you started with the CoreNLP client functionality.

{% include alerts.html %}
{{note}}
{{"It is highly advised to start the server in a context manager (e.g. `with CoreNLPClient(...) as client:`) to ensure
the server is properly shut down when your Python application finishes." | markdownify}}
{{end}}


### Importing the client

Importing the client from Stanza is as simple as a one-liner:

```python
from stanza.server import CoreNLPClient
```

### Starting a client-server communication and running annotation

Here we are going to run CoreNLP annotation on some example sentences. We start by first instantiating a `CoreNLPClient` object, and then pass the text into the client with the `annotate` function. Note that here we use the recommended Python `with` statement to start the client, which can make sure that the client and server are properly closed after the annotation:

```python
text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], 
        timeout=30000,
        memory='16G') as client:
    ann = client.annotate(text)
```

The CoreNLP server will be automatically started in the background upon the instantiation of the client, so normally you don't need to worry about it.

### Accessing basic annotation results

The following code shows how to manipulate the returned annotation object such that the sentences, tokens and various annotations can be accessed:

```python
    # get the first sentence
    sentence = ann.sentence[0]

    # get the constituency parse of the first sentence
    constituency_parse = sentence.parseTree
    print(constituency_parse)

    # get the first subtree of the constituency parse and its value
    print(constituency_parse.child[0])
    print(constituency_parse.child[0].value)
    
    # get the dependency parse of the first sentence
    dependency_parse = sentence.basicDependencies
    print(dependency_parse)

    # get an entity mention from the first sentence
    print(sentence.mentions[0])

    # get the first token of the first sentence
    token = sentence.token[0]
    print(token)

    # get the part-of-speech tag
    token.pos
    print(token.pos)

    # get the named entity tag
    print(token.ner)

    # access the coref chain in the input text
    print(ann.corefChain)
```

### Using Tokensregex, Semgrex and Tregex with the client

Separate client functions are provided to run [Tokensregex](https://nlp.stanford.edu/software/tokensregex.html), [Semgrex](https://nlp.stanford.edu/software/tregex.html), Tregex pattern matching with the CoreNLP client. The following example shows how to start a new client and use these three pattern matching functions:

```python
text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse'], 
        timeout=30000,
        memory='16G') as client:
    
    # Use tokensregex patterns to find who wrote a sentence.
    pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
    matches = client.tokensregex(text, pattern)
    # sentences contains a list with matches for each sentence.
    assert len(matches["sentences"]) == 3
    # length tells you whether or not there are any matches in this
    assert matches["sentences"][1]["length"] == 1
    # You can access matches like most regex groups.
    matches["sentences"][1]["0"]["text"] == "Chris wrote a simple sentence"
    matches["sentences"][1]["0"]["1"]["text"] == "Chris"

    # Use semgrex patterns to directly find who wrote what.
    pattern = '{word:wrote} >nsubj {}=subject >dobj {}=object'
    matches = client.semgrex(text, pattern)
    # sentences contains a list with matches for each sentence.
    assert len(matches["sentences"]) == 3
    # length tells you whether or not there are any matches in this
    assert matches["sentences"][1]["length"] == 1
    # You can access matches like most regex groups.
    matches["sentences"][1]["0"]["text"] == "wrote"
    matches["sentences"][1]["0"]["$subject"]["text"] == "Chris"
    matches["sentences"][1]["0"]["$object"]["text"] == "sentence"

    # Tregex example
    pattern = 'NP'
    matches = client.tregex(text, pattern)
    for match in matches:
        print(matches)
```