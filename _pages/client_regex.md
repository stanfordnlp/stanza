---
layout: page
title: Client Regex Usage
keywords: CoreNLP, client, regex, semgrex, tregex, tokensregex, usage
permalink: '/client_regex.html'
nav_order: 3
parent: Stanford CoreNLP Client
# toc: false
---

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
    pattern = '{word:wrote} >nsubj {}=subject >obj {}=object'
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

## Using Semgrex on depparse

New in v1.1
{: .label .label-green }

Note that each of the previous methods rely on using CoreNLP for the
language processing.  It is also possible to use semgrex to search
dependencies produced by the depparse module of stanza.

For example:

```python
import stanza
import stanza.server.semgrex as semgrex

# stanza.download("en")
nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

doc = nlp("Banning opal removed all artifact decks from the meta.  I miss playing lantern.")
semgrex_results = semgrex.process_doc(doc,
                                      "{pos:NN}=object <obl {}=action",
                                      "{cpos:NOUN}=thing <obj {cpos:VERB}=action")
print(semgrex_results)
```

This uses CoreNLP semgrex by launching a new java process.  Even
though it is inexpensive to launch a single process, it is best to
group all of the semgrex patterns to be run on a single doc into a
single function call.

The result will be a `CoreNLP_pb2.SemgrexResponse` protobuf object,
which contains nested lists for each sentence, for each semgrex query.
For this snippet, the result will look like:

```
result {
  result {
    match {
      index: 9
      node {
        name: "action"
        index: 3
      }
      node {
        name: "object"
        index: 9
      }
    }
  }
  result {
    match {
      index: 6
      node {
        name: "action"
        index: 3
      }
      node {
        name: "thing"
        index: 6
      }
    }
    match {
      index: 2
      node {
        name: "action"
        index: 1
      }
      node {
        name: "thing"
        index: 2
      }
    }
  }
}
result {
  result {
  }
  result {
    match {
      index: 4
      node {
        name: "action"
        index: 3
      }
      node {
        name: "thing"
        index: 4
      }
    }
  }
}
```

