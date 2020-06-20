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


## Importing the client

Importing the client from Stanza is as simple as a one-liner:

```python
from stanza.server import CoreNLPClient
```

## Starting a client-server communication and running annotation

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

## Accessing basic annotation results

The returned annotation object contains various annotations for sentences, tokens, and the entire document that can be accessed as native Python objects. For instance, the following code shows how to access various syntactic information of the first sentence in the piece of text in our example above:

```python
# get the first sentence
sentence = ann.sentence[0]

# get the constituency parse of the first sentence
constituency_parse = sentence.parseTree
print(constituency_parse)
```

This prints the constituency parse of the sentence, where the first child and its value can be accessed through `constituency_parse.child[0]` and `constituency_parse.child[0].value`, respectively

```
child {
  child {
    child {
      child {
        value: "Chris"
      }
      value: "NNP"
      score: -9.281864166259766
    }
    ...
  }
  ...
  value: "S"
  score: -50.052059173583984
}
value: "ROOT"
score: -50.20326614379883
```
{: .code-output }

Similarly, we can access the dependency parse of the first sentence as follows

```python
print(sentence.basicDependencies)
```
which prints output like the following

```
node {
  sentenceIndex: 0
  index: 1
}
...
edge {
  source: 2
  target: 1
  dep: "compound"
  isExtra: false
  sourceCopy: 0
  targetCopy: 0
  language: UniversalEnglish
}
...
root: 6
```
{: .code-output }

Here is an example to access token information, where we inspect the textual value of the token, its part-of-speech tag and named entity tag

```python
# get the first token of the first sentence
token = sentence.token[0]
print(token.value, token.pos, token.ner)
```
```
Chris NNP PERSON
```
{: .code-output }


Last but not least, we can examine the entity mentions in the first sentence and the coreference chain in the input text as follows

```python
# get an entity mention from the first sentence
print(sentence.mentions[0].entityMentionText)

# access the coref chain in the input text
print(ann.corefChain)
```
This gives us the mention text of the first entity mention in the first sentence, as well as a coref chain between entity mentions in the original text (the three mentions are "Chris Manning", "Chris", and "He", respectively, where CoreNLP has identified "Chris Manning" as the canonical mention of the cluster)
```
Chris Manning
[
mention {
  mentionID: 0
  mentionType: "PROPER"
  ...
}
mention {
  mentionID: 2
  mentionType: "PROPER"
  ...
}
mention {
  mentionID: 5
  mentionType: "PRONOMINAL"
  ...
}
representative: 0
...
]
```
{: .code-output}

## Using Tokensregex, Semgrex and Tregex with the client

Separate client functions are provided to run [Tokensregex](https://nlp.stanford.edu/software/tokensregex.html), [Semgrex](https://nlp.stanford.edu/software/tregex.html), Tregex pattern matching with the CoreNLP client. The following example shows how to start a new client and use TokensRegex to find patterns in text:

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
    print(len(matches["sentences"])) # prints: 3
    # length tells you whether or not there are any matches in this
    print(matches["sentences"][1]["length"]) # prints: 1
    # You can access matches like most regex groups.
    print(matches["sentences"][1]["0"]["text"]) # prints: "Chris wrote a simple sentence"
    print(matches["sentences"][1]["0"]["1"]["text"]) # prints: "Chris"
```

Aside from surface level patterns, the `CoreNLPClient` also allows you to use CoreNLP to extract patterns in syntactic structures. Here is an example shows how to use Semgrex and Tregex on the same piece of text:

```python
text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse'],
        timeout=30000,
        memory='16G') as client:

    # Use semgrex patterns to directly find who wrote what.
    pattern = '{word:wrote} >nsubj {}=subject >dobj {}=object'
    matches = client.semgrex(text, pattern)
    # sentences contains a list with matches for each sentence.
    print(len(matches["sentences"])) # prints: 3
    # length tells you whether or not there are any matches in this
    print(matches["sentences"][1]["length"]) # prints: 1
    # You can access matches like most regex groups.
    print(matches["sentences"][1]["0"]["text"]) # prints: "wrote"
    print(matches["sentences"][1]["0"]["$subject"]["text"]) # prints: "Chris"
    print(matches["sentences"][1]["0"]["$object"]["text"]) # prints: "sentence"

    # Tregex example
    pattern = 'NP'
    matches = client.tregex(text, pattern)
    # You can access matches similarly
    print(matches['sentences'][1]['1']['match']) # prints: "(NP (DT a) (JJ simple) (NN sentence))\n"
```
