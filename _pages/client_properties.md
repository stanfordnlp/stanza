---
layout: page
title: Advanced Usage & Client Customization
keywords: CoreNLP, client, properties, server start, advanced usage, customization
permalink: '/client_properties.html'
nav_order: 3
parent: Stanford CoreNLP Client
toc: true
---

In this section, we introduce how to customize the client options such that you can annotate a different language, use a different CoreNLP model, or have finer control over how you want the CoreNLP client or server to start.

## Switching Language

Stanza by default starts an English CoreNLP pipeline when a client is initialized. You can switch to a different language by setting a simple `properties` argument when the client is initialized. The following example shows how to start a client with default French models:

```python
with CoreNLPClient(properties='french') as client:
```

Alternatively, you can also use the [ISO 639-1 code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) for a language:
```python
with CoreNLPClient(properties='fr') as client:
```

This will initialize a `CoreNLPClient` object with the default set of French models. If you want to further customize the models used by the CoreNLP server, please read on.

{% include alerts.html %}
{{ note }}
{{ "Currently CoreNLP only provide official support for 6 human languages. For a full list of languages and models available, please see [the CoreNLP website](https://stanfordnlp.github.io/CoreNLP/index.html#human-languages-supported)." | markdownify }}
{{ end }}


## Using Customized Models by Setting Client Properties

Without further customization, a background CoreNLP server will use a default list of models for a language. This usually works very well out-of-the-box. However, sometimes it becomes useful to customize the models used by the CoreNLP server. For example, you might want to use a dictionary-based NER model instead of a statistical one, or you might want to switch to using the PCFG parser instead of the default shift-reduce parser.

Similar to switching languages, setting models used by the server can be done again via the `properties` argument when initializing the `CoreNLPClient` object. This argument can take three types of values:

| Properties Type | Example | Description |
| --- | --- | --- |
| Stanford CoreNLP supported language | french | One of {arabic, chinese, english, french, german, spanish}, this will use Stanford CoreNLP defaults for that language |
| Python dictionary | {'annotators': 'tokenize,ssplit,pos', 'pos.model': '/path/to/custom-model.ser.gz'} | A Python dictionary specifying the properties, the properties will be written to a tmp file |
| File path | /path/to/server.props | Path on the file system or CLASSPATH to a properties file |

Below are examples that illustrate how to use the three different types of `properties`:

- Using a language name:
```python
with CoreNLPClient(properties='french') as client:
```
As introduced above, this option allows quick switch between languages, and a default list of models will be used for each language.

- Using a Python dictionary
```python
with CoreNLPClient(properties={
        'annotators': 'tokenize,ssplit,pos',
        'pos.model': '/path/to/custom-model.ser.gz'
    }) as client:
```
This option allows you to override the default models used by the server, by providing (model name, model path) pairs.

- Using a properties file:
```python
with CoreNLPClient(properties='/path/to/server.props') as client:
```
This option allows the finest level of control over what annotators and models are going to be used in the server. For details on how to write a property file, please see the [instructions on configuring CoreNLP property files](https://stanfordnlp.github.io/CoreNLP/cmdline.html#configuring-corenlp-properties).


## Commonly-used CoreNLP Client Options

Using customized `properties` is the most common way to customize your CoreNLP server. However, sometimes you may want to have even more control on different aspects of your client-server, such as the port that the client uses to communicate with the server, the output format from the server, or the memory that you want to allocate to your server.

Here we provide a list of commonly-used arguments that you can initialize your `CoreNLPClient` with, along with their default values and descriptions:

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| annotators | str | "tokenize,<wbr>ssplit,<wbr>lemma,<wbr>pos,<wbr>ner,<wbr>depparse" | The default list of CoreNLP annotators the server will use |
| properties | - | None | See "Setting Client Properties" section above |
| endpoint | str | http://localhost:9000 | The host and port where the CoreNLP server will run on; change this when the default port 9000 is occupied. |
| classpath | str | None | Classpath to use for CoreNLP.  None means using the classpath as set by the `$CORENLP_HOME` environment variable, "$CLASSPATH" means to use the system CLASSPATH, and otherwise, the given string is used |
| timeout | int | 15000 | The maximum amount of time, in milliseconds, to wait for an annotation to finish before cancelling it. |
| threads | int | 5 | The number of threads to hit the server with. If, for example, the server is running on an 8 core machine, you can specify this to be 8, and the client will allow you to make 8 simultaneous requests to the server. |
| output_format | str | "serialized" | The default output format to use for the server response, unless otherwise specified. If set to be "serialized", the response will be converted to local Python objects (see usage examples [here](client_usage.md)). For a list of all supported output format, see the [CoreNLP output options page](https://stanfordnlp.github.io/CoreNLP/cmdline.html). |
| memory | str | "4G" | This specifies the memory used by the CoreNLP server process. |
| start_server | bool | True | Whether to start the CoreNLP server when initializing the Python `CoreNLPClient` object. By default the CoreNLP server will be started using the provided options. |
| stdout | file | sys.stdout | The standard output used by the CoreNLP server process. |
| stderr | file | sys.stderr | The standard error used by the CoreNLP server process. |
| be_quiet | bool | True | If set to False, the server process will print detailed error logs. Useful for diagnosing errors. |
| max_char_length | int | 100000 | The max number of characters that will be accepted and processed by the CoreNLP server in a single request. |
| preload | bool | True | Load the annotators immediately upon server start; otherwise the annotators will be lazily loaded upon the first annotation request is made. |

Here is a quick example that specifies a list of annotators to load, allocates 8G of memory to the server, uses plain text output format, and requests the server to print detailed error logs during annotation:

```python
with CoreNLPClient(
    annotators='tokenize,ssplit,pos,lemma,ner',
    output_format='text',
    memory='8G',
    be_quiet=False) as client:
```

### Using a CoreNLP server on a remote machine

With the endpoint option, you can even connect to a remote CoreNLP server running in a different machine:
```python
with CoreNLPClient(endpoint='http://remote-server-address:9000') as client:
```


## More Advanced CoreNLP Server Settings

Apart from the above options, there are some very advanced settings that you may need to customize how the CoreNLP server will start in the background. They are summarized in the following table:

| Option |  Description |
| --- | --- |
| server_id | ID for the server, label attached to server's shutdown key file |
| status_port | Port to server status check endpoints |
| uriContext | URI context for server |
| strict | Obey strict HTTP standards |
| ssl | If true, start server with (an insecure) SSL connection |
| key | .jks file to load if ssl is enabled |
| username | The username component of a username/password basic auth credential |
| password | The password component of a username/password basic auth credential |
| blacklist | a list of IPv4 addresses to ban from using the server |

You can also find more documention for the server's start up options on [the CoreNLP Server website](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html).

Here we highlight two common use cases on why you may need these options.

### Changing server ID when using multiple CoreNLP servers on a machine

When a CoreNLP server is started, it will write a special shutdown key file to the local disk, to indicate its running status.
This will create an issue when multiple servers need to be run simultaneously on a single machine, since a second server won't be able to write and delete its own shutdown key file.
This is easily solvable by giving a special server ID to the second server instance, when the client is initialized:
```python
with CoreNLPClient(server_id='second-server-name') as client:
```

### Protecting a CoreNLP server with password

You can even password-protect a CoreNLP server process, so that other users on the same machine won't be able to access or change your CoreNLP server:
```python
with CoreNLPClient(username='myusername', password='1234') as client:
```

Now you'll need to provide the same username and password when you call the `annotate` function of the client, so that the request can authenticate itself with the server:
```python
ann = client.annotate(text, username='myusername', password='1234')
```

Easy, right?

## Dynamically Changing Properties for Each Annotation Request

Properties for the CoreNLP pipeline run on text can be set for each particular annotation request.
If properties are set for a particular request, the server's initialization properties will be overridden.
This allows you to dynamically change your annotation need, without needing to start a new client-server from scratch.

Request-level properties can be registered with Stanza's `CoreNLPClient` to maximize efficiency.
Upon registration, the client will maintain a `properties_cache` to map keys to particular property settings.
Alternatively, request-level properties can be specified as a Stanford CoreNLP support language
to use the language defaults, or a full Python dictionary for maximal flexibility.

Here is an example of how to register a set of properties with the client's `properties_cache`,
and how to use those properties via the key for annotation:
```python
FRENCH_CUSTOM_PROPS = {
    'annotators': 'tokenize,ssplit,pos,parse', 'tokenize.language': 'fr',
    'pos.model': 'edu/stanford/nlp/models/pos-tagger/french/french.tagger',
    'parse.model': 'edu/stanford/nlp/models/lexparser/frenchFactored.ser.gz',
    'outputFormat': 'text'
}

with CoreNLPClient(annotators='tokenize,ssplit,pos') as client:
    client.register_properties_key('fr-custom', FRENCH_CUSTOM_PROPS)
    ann = client.annotate(text, properties_key='fr-custom')
```

Alternatively, request-level properties can simply be a language that you want to run the CoreNLP pipeline for:
```python
ann = client.annotate(text, properties='german')
```

Or, a dictionary that specifies all properties you want to set/override:
```python
ann = client.annotate(text, properties=FRENCH_CUSTOM_PROPS)
```

Similarly to `CoreNLPClient` initialization, you can also specify the annotators and output format for CoreNLP for individual annotation requests as:
```python
ann = client.annotate(text, properties=FRENCH_CUSTOM_PROPS, annotators='tokenize,ssplit,pos', output_format='json')
```

<!-- There are a variety of ways to set properties for the server at start time.  Also, each request can have properties.
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
or set specific properties for requests. -->
