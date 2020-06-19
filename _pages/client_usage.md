---
layout: page
title: Client Basic Usage
keywords: CoreNLP, client, usage
permalink: '/client_usage.html'
nav_order: 2
parent: Stanford CoreNLP Client
toc: false
---

After CoreNLP has been properly set up, you can start up the server and make requests in Python code with Stanza's help.
Below is a comprehensive example of starting a server, making requests, and accessing data from the returned Document object. We have also prepared a [comprehensive Jupyter notebook tutorial](https://github.com/stanfordnlp/stanza/blob/master/demo/StanfordNLP_CoreNLP_Interface.ipynb), which you can experiment with interactively.
By default, CoreNLP Client uses `protobuf` for message passing. A full definition of our protocols (a.k.a., our supported annotations) can be found [here](https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/pipeline/CoreNLP.proto).

{% include alerts.html %}
{{note}}
{{"It is highly advised to start the server in a context manager (e.g. `with CoreNLPClient(...) as client:`) to ensure
the server is properly shut down when your Python application finishes." | markdownify}}
{{end}}

```python
from stanza.server import CoreNLPClient

# example text
print('---')
print('input text')
print('')

text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."

print(text)

# set up the client
print('---')
print('starting up Java Stanford CoreNLP Server...')

# set up the client
with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], timeout=30000, memory='16G') as client:
    # submit the request to the server
    ann = client.annotate(text)

    # get the first sentence
    sentence = ann.sentence[0]

    # get the constituency parse of the first sentence
    print('---')
    print('constituency parse of first sentence')
    constituency_parse = sentence.parseTree
    print(constituency_parse)

    # get the first subtree of the constituency parse
    print('---')
    print('first subtree of constituency parse')
    print(constituency_parse.child[0])

    # get the value of the first subtree
    print('---')
    print('value of first subtree of constituency parse')
    print(constituency_parse.child[0].value)

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