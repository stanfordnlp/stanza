---
layout: page
title: Client Regex Usage
keywords: CoreNLP, client, regex, semgrex, tregex, tokensregex, usage
permalink: '/client_regex.html'
nav_order: 2.5
parent: Stanford CoreNLP Client
toc: false
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

