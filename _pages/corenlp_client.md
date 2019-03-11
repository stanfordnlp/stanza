---
title: Stanford CoreNLP Client
keywords: StanfordNLP, Stanford CoreNLP, Client, Server, Python
permalink: '/corenlp_client.html'
---

## Overview

StanfordNLP allows users to access our Java toolkit Stanford CoreNLP via a server interface.  Once the Java server is activated, requests can be made in Python, and a Document-like object will be returned.  You can find out more info about the full functionality of Stanford CoreNLP [here](https://stanfordnlp.github.io/CoreNLP/).

## Setup

* Download the latest version of Stanford CoreNLP and models jars from [here](https://stanfordnlp.github.io/CoreNLP/download.html)
* Store the models jar in the distribution folder
* Set the `CORENLP_HOME` environment variable to the location of the folder.  Example: `export CORENLP_HOME=/path/to/stanford-corenlp-full-2018-10-05`
 
## Usage

After the above steps have been taken, you can start up the server and make requests in Python code.
Below is a comprehensive example of starting a server, making requests, and accessing data from the returned object.

```python
from stanfordnlp.server import CoreNLPClient

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
```

## CoreNLP Client Options
During initialization, a `CoreNLPClient` object accepts the following options as arguments:

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| annotators | list | ["tokenize", "ssplit", "lemma", "pos", "ner", "depparse"] | The default list of CoreNLP annotators to run for a request, unless otherwise specified in the call to `annotate()`. |
| properties | dict | `empty` | This allows you to specify the detailed CoreNLP annotation properties as a Python `dict` object. For example, setting `{"tokenize.language": "en", "tokenize.options": "ptb3Escaping=false"}` will specify an English tokenizer and disable PTB escaping. For all available properties, please see the [CoreNLP annotators page](https://stanfordnlp.github.io/CoreNLP/annotators.html). |
| endpoint | str | http://localhost:9000 | The host and port where the CoreNLP server will run on. |
| timeout | int | 15000 | The maximum amount of time, in milliseconds, to wait for an annotation to finish before cancelling it. |
| threads | int | 5 | The number of threads to hit the server with. If, for example, the server is running on an 8 core machine, you can specify this to be 8, and the client will allow you to make 8 simultaneous requests to the server. |
| output_format | str | "serialized" | The default output format to use for the server response, unless otherwise specified. If set to be "serialized", the response will be converted to local Python objects (see usage examples above). For a list of all supported output format, see the [CoreNLP output options page](https://stanfordnlp.github.io/CoreNLP/cmdline.html). |
| memory | str | "4G" | This specifies the memory used by the CoreNLP server process. |
| start_server | bool | True | Whether to start the CoreNLP server when initializing the Python `CoreNLPClient` object. By default the CoreNLP server will be started using the provided options. |
| stdout | file | sys.stdout | The standard output used by the CoreNLP server process. |
| stderr | file | sys.stderr | The standard error used by the CoreNLP server process. |
| be_quiet | bool | True | If set to False, the server process will print detailed error logs. Useful for diagnosing errors. |
| max_char_length | int | 100000 | The max number of characters that will be accepted and processed by the CoreNLP server in a single request. |
