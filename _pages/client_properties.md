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

## Overview

By default, the CoreNLP server will run the following English annotators:

```
tokenize,ssplit,pos,lemma,ner,depparse,coref,kbp
```

There are a variety of ways to customize a CoreNLP pipeline, including:

* using a different list of annotators (e.g. tokenize,ssplit,pos)
* processing a different language (e.g. French)
* using custom models (e.g. my-custom-depparse.gz)
* returning different output formats (e.g. JSON)

These customizations are achieved by specifying properties.

The first step is always importing `CoreNLPClient`

```python
from stanza.server import CoreNLPClient
```

When starting a CoreNLP server via Stanza, a user can choose what
properties to initialize the server with. For instance, here is
an example of launching a server with a different parser model
that returns JSON:

```python
CUSTOM_PROPS = {"parse.model": "edu/stanford/nlp/models/srparser/englishSR.beam.ser.gz"}

with CoreNLPClient(properties=CUSTOM_PROPS, output_format="json") as client:
```

Or one could launch a server with CoreNLP French defaults as in this example:

```python
with CoreNLPClient(properties="french") as client:
```

When communicating with a CoreNLP server via Stanza, a user can send specific
properties for one time use with that request. These request level properties
allow for a dynamic NLP application which can apply different pipelines 
depending on input text.

For instance, one could switch between German and French pipelines:

```python
french_text = "Emmanuel Macron est le président de la France."
german_text = "Angela Merkel ist die deutsche Bundeskanzlerin."

with CoreNLPClient() as client:
    french_ann = client.annotate(french_text, properties="fr")
    german_ann = client.annotate(german_text, properties="de")
```

Or move between custom biomedical and financial text processing pipelines:

```python
BIOMEDICAL_PROPS = {"depparse.model": "/path/to/biomedical-parser.gz"}
FINANCE_PROPS = {"depparse.model": "/path/to/finance-parser.gz"}

with CoreNLPClient() as client:
    bio_ann = client.annotate(bio_text, properties=BIOMEDICAL_PROPS)
    finance_ann = client.annotate(finance_text, properties=FINANCE_PROPS)
```

## CoreNLP Server Start Options (Pipeline)

There are three ways to specify pipeline properties when starting a CoreNLP server:

| Properties Type | Example | Description |
| --- | --- | --- |
| Stanford CoreNLP supported language | french | One of {arabic, chinese, english, french, german, spanish} (or the ISO 639-1 code), this will use Stanford CoreNLP defaults for that language |
| Python dictionary | {'annotators': 'tokenize,ssplit,pos', 'pos.model': '/path/to/custom-model.ser.gz'} | A Python dictionary specifying the properties, the properties will be written to a tmp file |
| File path | /path/to/server.props | Path on the file system or CLASSPATH to a properties file |

For convenience one can also specify the list of `annotators` and the desired `output_format` in the `CoreNLPClient` constructor.
The values for those two arguments will override any additional properties supplied at construction time.

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

For convenience one can also specify the list of `annotators` and the desired `output_format` in the `CoreNLPClient` constructor.

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| annotators | str | "tokenize,<wbr>ssplit,<wbr>lemma,<wbr>pos,<wbr>ner,<wbr>depparse" | The default list of CoreNLP annotators the server will use |
| output_format | str | "serialized" | The default output format to use for the server response, unless otherwise specified. If set to be "serialized", the response will be converted to local Python objects (see usage examples [here](client_usage.md)).  |

The values for those two arguments will override any additional properties supplied at construction time.

```python
with CoreNLPClient(properties='french', annotators='tokenize,ssplit,mwt,pos,ner,parse', output_format='json') as client:
```

## CoreNLP Server Start Options (Server)

In addition to customizing the pipeline the server will run, a variety of
server specific properties can be specified at server construction time.

Here we provide a list of commonly-used arguments that you can initialize your `CoreNLPClient` with, along with their default values and descriptions:

| Option name | Type | Default | Description |
| --- | --- | --- | --- |
| endpoint | str | http://localhost:9000 | The host and port where the CoreNLP server will run on; change this when the default port 9000 is occupied. |
| classpath | str | None | Classpath to use for CoreNLP.  None means using the classpath as set by the `$CORENLP_HOME` environment variable, "$CLASSPATH" means to use the system CLASSPATH, and otherwise, the given string is used |
| timeout | int | 60000 | The maximum amount of time, in milliseconds, to wait for an annotation to finish before cancelling it. |
| threads | int | 5 | The number of threads to hit the server with. If, for example, the server is running on an 8 core machine, you can specify this to be 8, and the client will allow you to make 8 simultaneous requests to the server. |
| memory | str | "5G" | This specifies the memory used by the CoreNLP server process. |
| start_server | stanza.<wbr>server.<wbr>StartServer | FORCE_START | Whether to start the CoreNLP server when initializing the Python `CoreNLPClient` object. By default the CoreNLP server will be started using the provided options. Alternatively, `DONT_START` doesn't start a new CoreNLP server and attempts to connect to an existing server instance at `endpoint`; `TRY_START` tries to start a new server instance at the endpoint provided, but doesn't fail like `FORCE_START` if one is already running there. Note that this Enum is new in Stanza v1.1, and in previous versions it only supports boolean input. |
| stdout | file | sys.stdout | The standard output used by the CoreNLP server process. |
| stderr | file | sys.stderr | The standard error used by the CoreNLP server process. |
| be_quiet | bool | False | If set to False, the server process will print detailed error logs. Useful for diagnosing errors. |
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
{% include alerts.html %}
{{ note }}
{{ "The be_quiet option is set to False by default! It is advised to review CoreNLP server logs when starting out to make sure any errors are not happening on the server side of your application. If your application is generally stable, you can set be_quiet=True to stop seeing CoreNLP server log output." | markdownify }}
{{ end }}



## CoreNLP Server Start Options (Advanced)

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
| blockList | a list of IPv4 addresses to ban from using the server |

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

## Switching Languages

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
{{ "Currently CoreNLP only provides official support for 6 human languages. For a full list of languages and models available, please see [the CoreNLP website](https://stanfordnlp.github.io/CoreNLP/index.html#human-languages-supported)." | markdownify }}
{{ end }}

## Using a CoreNLP server on a remote machine

With the endpoint option, you can even connect to a remote CoreNLP server running in a different machine:
```python
with CoreNLPClient(endpoint='http://remote-server-address:9000') as client:
```

## Dynamically Changing Properties for Each Annotation Request

Properties for the CoreNLP pipeline run on text can be set for each particular annotation request.
If properties are set for a particular request, the server's initialization properties will be overridden.
This allows you to dynamically change your annotation need, without needing to start a new client-server from scratch.

Request level properties can be specified with a Python dictionary, or the name of a CoreNLP supported language.

Here is an example of making a request with a custom dictionary of properties:

```python
FRENCH_CUSTOM_PROPS = {
    'annotators': 'tokenize,ssplit,pos,parse', 'tokenize.language': 'fr',
    'pos.model': 'edu/stanford/nlp/models/pos-tagger/french/french.tagger',
    'parse.model': 'edu/stanford/nlp/models/lexparser/frenchFactored.ser.gz',
    'outputFormat': 'text'
}

with CoreNLPClient() as client:
    ann = client.annotate(text, properties=FRENCH_CUSTOM_PROPS)
```

Alternatively, request-level properties can simply be a language that you want to run the CoreNLP pipeline for:
```python
ann = client.annotate(text, properties='german')
```

{% include alerts.html %}
{{ note }}
{{ "A subtle point to note is that when requests are sent with custom properties, those custom properties will overwrite the properties the server was started with, unless a CoreNLP language name is specified, in which case the server start properties will be ignored and the CoreNLP defaults for that language will be written on top of the original CoreNLP defaults." | markdownify }}
{{ end }}

Similarly to `CoreNLPClient` initialization, you can also specify the annotators and output format for CoreNLP for individual annotation requests as:
```python
ann = client.annotate(text, properties=FRENCH_CUSTOM_PROPS, annotators='tokenize,ssplit,pos', output_format='json')
```

