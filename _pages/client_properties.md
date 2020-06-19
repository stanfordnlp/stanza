---
layout: page
title: Customizing Client Properties
keywords: CoreNLP, client, properties, server start
permalink: '/client_properties.html'
nav_order: 3
parent: Stanford CoreNLP Client
toc: true
---

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


### CoreNLP Client Options
During initialization, a `CoreNLPClient` object accepts the following options as arguments:

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| annotators | str | "tokenize,<wbr>ssplit,<wbr>lemma,<wbr>pos,<wbr>ner,<wbr>depparse" | The default list of CoreNLP annotators the server will use |
| properties | - | None | See "Customize Server Start" section below |
| endpoint | str | http://localhost:9000 | The host and port where the CoreNLP server will run on. |
| classpath | str | None | Classpath to use for CoreNLP.  None means $CORENLP_HOME, $CLASSPATH means to use the system $CLASSPATH, and otherwise, the given string is used |
| timeout | int | 15000 | The maximum amount of time, in milliseconds, to wait for an annotation to finish before cancelling it. |
| threads | int | 5 | The number of threads to hit the server with. If, for example, the server is running on an 8 core machine, you can specify this to be 8, and the client will allow you to make 8 simultaneous requests to the server. |
| output_format | str | "serialized" | The default output format to use for the server response, unless otherwise specified. If set to be "serialized", the response will be converted to local Python objects (see usage examples [here](client_usage.md)). For a list of all supported output format, see the [CoreNLP output options page](https://stanfordnlp.github.io/CoreNLP/cmdline.html). |
| memory | str | "4G" | This specifies the memory used by the CoreNLP server process. |
| start_server | bool | True | Whether to start the CoreNLP server when initializing the Python `CoreNLPClient` object. By default the CoreNLP server will be started using the provided options. |
| stdout | file | sys.stdout | The standard output used by the CoreNLP server process. |
| stderr | file | sys.stderr | The standard error used by the CoreNLP server process. |
| be_quiet | bool | True | If set to False, the server process will print detailed error logs. Useful for diagnosing errors. |
| max_char_length | int | 100000 | The max number of characters that will be accepted and processed by the CoreNLP server in a single request. |
| preload | bool | True | Load the annotators immediately upon server start |

Here is a quick example to specify which annotators to load and what output format to use

```python
with CoreNLPClient(annotators='tokenize,ssplit,pos,lemma,ner', output_format='text') as client:
```

### Customizing Server Start

When a CoreNLP Server is started it contains a set of default properties that define its default pipeline.
This is the pipeline that will run unless the requests specify otherwise through request properties.

The following values can be supplied to the `properties` argument for `CoreNLPClient`'s `init` method to build a default pipeline:

| Option | Example | Description |
| --- | --- | --- |
| file path | /path/to/server.props | Path on the file system or CLASSPATH to a properties file |
| Stanford CoreNLP supported language | french | One of {arabic, chinese, english, french, german, spanish}, this will use Stanford CoreNLP defaults for that language |
| Python dictionary | {'annotators': 'tokenize,ssplit,pos', 'pos.model': '/path/to/custom-model.ser.gz'} | A Python dictionary specifying the properties, the properties will be written to a tmp file |

If not using the file path or language name option, one can also specify which `annotators` to use and the desired `outputFormat` with the
`annotators` and `output_format` args to `CoreNLPClient`'s `init` method.

Below are code examples to illustrate these different options:

You can start the server by specifying a properties file
```python
with CoreNLPClient(properties='/path/to/server.props') as client:
```

Or, specifying a Stanford CoreNLP supported language
```python
with CoreNLPClient(properties='french') as client:
```

Or, last but not least, specifying properties from a Python dictionary
```python
with CoreNLPClient(properties={'annotators': 'tokenize,ssplit,pos', 'pos.model': '/path/to/custom-model.ser.gz'}) as client:
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

Here is a quick example of specifying username and password for your server at launch
```python
with CoreNLPClient(username='myusername', password='1234') as client:
```

If your CoreNLP server is password protected, here's how you can supply that information to make sure
annotation goes smoothly
```python
ann = client.annotate(text, username='myusername', password='1234')
```

### Request Properties

Properties for the StanfordCoreNLP pipeline run on text can be set for each request.
If the request has properties, these will override the server's defaults.

Request properties can be registered with Stanza's `CoreNLPClient` to maximize efficiency.
The client maintains a `properties_cache` to map keys to particular property settings.
Alternatively, request properties can be specified as a Stanford CoreNLP support language
to use the language defaults, or a full Python dictionary for maximal flexibility.

The following code examples below show examples of specifying request properties.

Here is an example of how to register a set of properties with the client's `properties_cache`,
and how to use those properties via the key for annotation
```python
FRENCH_CUSTOM_PROPS = {'annotators': 'tokenize,ssplit,pos,parse', 'tokenize.language': 'fr',
                       'pos.model': 'edu/stanford/nlp/models/pos-tagger/french/french.tagger',
                       'parse.model': 'edu/stanford/nlp/models/lexparser/frenchFactored.ser.gz',
                       'outputFormat': 'text'}

with CoreNLPClient(annotators='tokenize,ssplit,pos') as client:
    client.register_properties_key('fr-custom', FRENCH_CUSTOM_PROPS)
    ann = client.annotate(text, properties_key='fr-custom')
```

Alternatively, request properties can simply be a language that you want to run the default CoreNLP
pipeline for
```python
ann = client.annotate(text, properties='german')
```

Or, a dictionary that specifies all properties you want to set/override
```python
ann = client.annotate(text, properties=FRENCH_CUSTOM_PROPS)
```

Similarly to `CoreNLPClient` initialization, you can also specify the annotators and output format
for CoreNLP for individual annotation requests
```python
ann = client.annotate(text, properties=FRENCH_CUSTOM_PROPS, annotators='tokenize,ssplit,pos', output_format='json')
```
