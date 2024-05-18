---
layout: page
title: Tsurgeon
keywords: CoreNLP, client, constituencies, tsurgeon, tregex
permalink: '/tsurgeon.html'
nav_order: 6
parent: Stanford CoreNLP Client
toc: true
---

## Introduction

Stanza comes with an interface to Tsurgeon, a constituency tree rewriting tool included with CoreNLP.

Here is a brief example of how to use the context window for Tsurgeon

## Code sample

```python
from stanza.models.constituency.tree_reader import read_trees, read_treebank
from stanza.server import tsurgeon

TREEBANK = """
( (IP-MAT (NP-SBJ (PRO-N Það-það))
          (BEPI er-vera)
          (ADVP (ADV eiginlega-eiginlega))
          (ADJP (NEG ekki-ekki) (ADJ-N hægt-hægur))
          (IP-INF (TO að-að) (VB lýsa-lýsa))
          (NP-OB1 (N-D tilfinningu$-tilfinning) (D-D $nni-hinn))
          (IP-INF (TO að-að) (VB fá-fá))
          (IP-INF (TO að-að) (VB taka-taka))
          (NP-OB1 (N-A þátt-þáttur))
          (PP (P í-í)
              (NP (D-D þessu-þessi)))
          (, ,-,)
          (VBPI segir-segja)
          (NP-SBJ (NPR-N Sverrir-sverrir) (NPR-N Ingi-ingi))
          (. .-.)))
"""

# expected result
#(ROOT
#  (IP-MAT
#    (NP-SBJ (PRO-N Það))
#    (BEPI er)
#    (ADVP (ADV eiginlega))
#    (ADJP (NEG ekki) (ADJ-N hægt))
#    (IP-INF (TO að) (VB lýsa))
#    (NP-OB1 (N-D tilfinningunni))
#    (IP-INF (TO að) (VB fá))
#    (IP-INF (TO að) (VB taka))
#    (NP-OB1 (N-A þátt))
#    (PP
#      (P í)
#      (NP (D-D þessu)))
#    (, ,)
#    (VBPI segir)
#    (NP-SBJ (NPR-N Sverrir) (NPR-N Ingi))
#    (. .)))


treebank = read_trees(TREEBANK)

with tsurgeon.Tsurgeon(classpath="$CLASSPATH") as tsurgeon_processor:
    form_tregex = "/^(.+)-.+$/#1%form=word !< __"
    form_tsurgeon = "relabel word /^.+$/%{form}/"

    noun_det_tregex = "/^N-/ < /^([^$]+)[$]$/#1%noun=noun $+ (/^D-/ < /^[$]([^$]+)$/#1%det=det)"
    noun_det_relabel = "relabel noun /^.+$/%{noun}%{det}/"
    noun_det_prune = "prune det"

    for tree in treebank:
        updated_tree = tsurgeon_processor.process(tree, (form_tregex, form_tsurgeon))[0]
        print("{:P}".format(updated_tree))
        updated_tree = tsurgeon_processor.process(updated_tree, (noun_det_tregex, noun_det_relabel, noun_det_prune))[0]
        print("{:P}".format(updated_tree))
```

In this example, we perform two operations on the tree: replace all
word nodes with the form, and squish the `N` and `D` nodes with `$` in
them into one node.  In this way, we can prepare the Icelandic
treebank for use in training a Stanza constituency parser.

## Details

Tsurgeon can operation on trees read from a file or text, such as in
this example, or on trees produced by the constituency parser.

The context window opens a pipe to a Java executable and then uses a
protobuf to communicate.  Java and CoreNLP both need to be available
on your system for this to work.

## Further information

More information on the tregex patterns used is available on the [Tregex Javadoc page](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/trees/tregex/TregexPattern.html)

There is also more documentation on [the Tsurgeon operations available](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/trees/tregex/tsurgeon/Tsurgeon.html)

## Standalone Tregex Operations

So far, we haven't found a use for Tregex by itself without Tsurgeon
from Python.  If you have such a use case, please contact us via
[github](https://github.com/stanfordnlp/stanza) and we will help work
out an interface for Tregex.  Likely candidates would be finding
specific subtrees or testing whether or not a tree matches an
expression.