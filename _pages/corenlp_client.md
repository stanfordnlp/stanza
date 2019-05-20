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
Below is a comprehensive example of starting a server, making requests, and accessing data from the returned Document object.

Note: It is highly advised to start the server in a context manager (e.g. `with CoreNLPClient(...) as client:`) to ensure
the server is properly shut down when your Python application finishes.

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

## Customizing Properties For Server Start And Requests

There are a variety of ways to set properties for the server at start time.  Also, each request can have properties.
For example, if your application may receive German or French input, you could start the server with the default
German properties and you could send requests for French text with the default French properties, and alternate
back and forth depending on the input text language.  You could also imagine having one pipeline

For context, the Java server takes in requests, runs a StanfordCoreNLP pipeline, and sends a response.
Annotators within a pipeline often use models to perform their work (e.g. a part-of-speech tagging model or
a dependency parsing model).  For efficiency, the server maintains two caches.  One for the models (so they only
have to be loaded once) and one for already built StanfordCoreNLP pipelines.  

The server actually maintains two caches, one for the models, and one for pre-built Stanford CoreNLP pipelines.
Loading models takes a significant amount of time (potentially on the order of a minute in some cases).  But
building a pipeline can also take some time (on the order of a half second).  So throughput would be significantly
impacted if pipelines were built per request.  The server will keep around pipelines with particular configurations
to avoid this penalty.  The first request using a particular model or pipeline configuration will have delay
associated with loading the model or building the pipeline.  But future requests should not have this delay.

To avoid the first request having a long delay, one can use the `preload` option to load a set of models when
the server launches.

Below are a collection of specific examples to demonstrate different ways to customize the server at start time
or set specific properties for requests.

### Customize Server Start

When a StanfordCoreNLP Server is started it contains a set of default properties that define it's default pipeline.
This is the pipeline that will run if no request properties are specified.

The following values can be supplied to the `properties` argument for `CoreNLPClient`'s `init` method:

| Option | Example | Description |
| --- | --- | --- |
| file path | /path/to/server.props | Path on the file system or CLASSPATH to a properties file |
| Stanford CoreNLP supported language | french | One of {arabic, chinese, english, french, german, spanish}, this will use Stanford CoreNLP defaults for that language |
| Python dictionary | {'annotators': 'tokenize,ssplit,pos', 'pos.model': '/path/to/custom-model.ser.gz'} | A Python dictionary specifying the properties, the properties will be written to a tmp file |

If not using the file path or language name option, one can also specify which `annotators` to use and the desired `outputFormat` with the
`annotators` and `output_format` args to `CoreNLPClient`'s `init` method.

Below are code examples to illustrate all of this:

#### Specify a properties file
```
with CoreNLPClient(properties='/path/to/server.props') as client:
```

#### Specify a Stanford CoreNLP supported language
```
with CoreNLPClient(properties='french') as client:
```

#### Specify a Python dictionary
```
with CoreNLPClient(properties={'annotators': 'tokenize,ssplit,pos', 'pos.model': '/path/to/custom-model.ser.gz'}) as client:
```

#### Specify annotators and output_format
```
with CoreNLPClient(annotators='tokenize,ssplit,pos,lemma,ner', output_format='text') as client:
```

### Stanford CoreNLP Server Settings
In addition to setting properties for the pipeline used by the Stanford CoreNLP server, there are also a variety of settings
for the server itself.  The table below describes these settings:

| Option |  Description |
| --- | --- |
| server_id | ID for the server, label attached to server's shutdown key file |
| status_port | port to server status check endpoints |
| uriContext | URI context for server |
| strict | obey strict HTTP standards |
| ssl | if true, start server with (an insecure) SSL connection |
| key | .jks file to load if ssl is enabled |
| username | The username component of a username/password basic auth credential |
| password | The password component of a username/password basic auth credential |
| preload | load this list of annotators immediately at server launch |
| blacklist | a list of IPv4 addresses to ban from using the server |

All of these options can be set with `CoreNLPClient`'s `init` method.

There is more documention [here](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html) for the server's start up options.

#### Specify username and password for your server
```
with CoreNLPClient(username='myusername', password='1234') as client:
```

### Request Properties

Properties for the StanfordCoreNLP pipeline run on text can be set for each request.
If the request has properties, these will override the server's defaults.

The client maintains a `properties_cache` to map keys to particular property settings.
Or a Stanford CoreNLP support language can be specified to use the language defaults, or
a full Python dictionary.

The following code examples below show examples of specifying request properties.

#### Register a set of properties with the client's properties_cache, use key
```
FRENCH_CUSTOM_PROPS = {'annotators': 'tokenize,ssplit,pos,parse', 'tokenize.language': 'fr',
                       'pos.model': 'edu/stanford/nlp/models/pos-tagger/french/french.tagger',
                       'parse.model': 'edu/stanford/nlp/models/lexparser/frenchFactored.ser.gz',
                       'outputFormat': 'text'}

with CoreNLPClient(annotators='tokenize,ssplit,pos') as client:
    client.register_properties_key('fr-custom', FRENCH_CUSTOM_PROPS)
    ann = client.annotate(text, properties_key='fr-custom')
```

#### Set request properties as a Python dictionary
```
ann = client.annotate(text, properties=FRENCH_CUSTOM_PROPS)
```

#### Specify a StanfordCoreNLP supported language
```
ann = client.annotate(text, properties='german')
```

#### Specify annotators and output_format
```
ann = client.annotate(text, properties=FRENCH_CUSTOM_PROPS, annotators='tokenize,ssplit,pos', output_format='json'
```

## CoreNLP Client Options
During initialization, a `CoreNLPClient` object accepts the following options as arguments:

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| annotators | str | "tokenize,ssplit,lemma,pos,ner,depparse" | The default list of CoreNLP annotators the server will use` |
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
