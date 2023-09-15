---
layout: page
title: Semgrex and Ssurgeon
keywords: CoreNLP, client, dependencies, semgrex, ssurgeon
permalink: '/semgrex.html'
nav_order: 5
parent: Stanford CoreNLP Client
toc: true
---



Here we present a complete example of using Semgrex and Ssurgeon to fix some errors in a UD treebank.

In particular, there are two classes of edits we want to make to
UD_English-LinES.  There are cases where `he's` is labeled as a `to be`
verb, even though it is definitely `he has`.  Also, other English UD
treebanks are marked with MWT on words such as `won't`, and we can do
that automatically with Ssurgeon.

To use this, you will need Semgrex >= 1.5.1 and CoreNLP >= 4.5.5.
(Certain components from the examples below were present in earlier versions, but the tools have been upgraded since their initial release.)

## Example search

To search for examples of the first one, we try a couple things.  The
first is that this often happens when there are two consecutive `to
be` verbs in a row, so we search for exactly that:

```
python3 stanza/server/semgrex.py  --input_file UD_English-LinES/en_lines-ud-train.conllu "{word:/'s/;lemma:be} . {lemma:be}" --matches_only
```

Here, `--matches_only` makes it only print out matching sentences.
Otherwise, Semgrex would print out the whole treebank, even though
there are only a few matching sentences.

This prints out sentences such as

```
# text = He's been waiting for you since eight o'clock.
# semgrex pattern |{word:/'s/;lemma:be} . {lemma:be}| matched at 2:'s
1       He      he      PRON    PERS-P3SG-NOM   Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs  4       nsubj   _       SpaceAfter=No
2       's      be      AUX     PRES    Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin   4       aux     _       _
3       been    be      AUX     PERF    Tense=Past|VerbForm=Part        4       aux     _       _
4       waiting wait    VERB    ING     Tense=Pres|VerbForm=Part        0       root    _       _


# sent_id = en_lines-ud-train-doc6-2341
# text = Rebecca's been to the Sputnik and she says it's terrific now.
# semgrex pattern |{word:/'s/;lemma:be} . {lemma:be}| matched at 2:'s
1       Rebecca Rebecca PROPN   SG-NOM  Number=Sing     3       nsubj   _       SpaceAfter=No
2       's      be      AUX     PRES-AUX        Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin   3       aux     _       _
3       been    be      AUX     PERF    Tense=Past|VerbForm=Part        0       root    _       _
4       to      to      ADP     _       _       6       case    _       _
5       the     the     DET     DEF     Definite=Def|PronType=Art       6       det     _       _
6       Sputnik Sputnik PROPN   SG-NOM  Number=Sing     3       obl     _       _
```

There are cases where there is an interposing word, though, and this
query does not find those.  We can observe that there are two cases we
might care about.  In the first sentence, `'s` and `been` are both
dependents of `waiting`.  We can try to find that as follows:

```
python3 stanza/server/semgrex.py  --input_file UD_English-LinES/en_lines-ud-train.conllu "{} > {word:/'s/;lemma:be}=has > {lemma:be}" --matches_only
```

Note that the `'s` word will be labeled as a named node in this query.
However, this query actually has an error.  Matching nodes is not
exclusive, meaning that the second `> {lemma:be}` can always match the
same node that matched `=has`.  We can fix that in a variety of ways.
One such solution is to mark both dependents, then enforce that they
are different:

```
python3 stanza/server/semgrex.py  --input_file UD_English-LinES/en_lines-ud-train.conllu "{} > {word:/'s/;lemma:be}=has > {lemma:be}=other : {}=has .. {}=other" --matches_only
```

The problem with this query is that there are cases with some spurious matches.  For example, in the dev set:

```
# sent_id = en_lines-ud-dev-doc6-3874
# text = If there's a father of the state, it's got to be him or no one.
# semgrex pattern |{} > {word:/'s/;lemma:be}=has > {lemma:be}=other : {}=has .. {}=other| matched at 15:him  has=3:'s other=11:'s
# semgrex pattern |{} > {word:/'s/;lemma:be}=has > {lemma:be}=other : {}=has .. {}=other| matched at 15:him  has=3:'s other=14:be
# semgrex pattern |{} > {word:/'s/;lemma:be}=has > {lemma:be}=other : {}=has .. {}=other| matched at 15:him  has=11:'s other=14:be
```

In this sentence, we actually would want to match `it has got to be ...`, so words `11` and `12`, not `11` and `14`.  The tag on `12` was
`AUX`, which we will use later

Furthermore, there is this sentence:

```
# text = Edward Shinza's one of the few who did his stretch and got his head split open ...
# semgrex pattern |{} > {word:/'s/;lemma:be}=has > {lemma:be}=other : {}=has .. {}=other| matched at 4:one  has=3:'s other=29:'s
```

In the test set, though, we find this sentence, which is exactly what we were looking for in terms of looking for words which skip a step.  We note that the tag for that word is `ADV`:

```
# sent_id = en_lines-ud-test-doc6-4988
# text = He's never been sent anywhere where there was anything left to do, he said.
# semgrex pattern |{} > {word:/'s/;lemma:be}=has > {lemma:be}=other : {}=has .. {}=other| matched at 5:sent  has=2:'s other=4:been
1       He      he      PRON    PERS-P3SG-NOM   Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs  5       nsubj:pass      _       SpaceAfter=No
2       's      be      AUX     PRES-AUX        Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin   5       aux:pass        _       _
3       never   never   ADV     NEG     _       5       advmod  _       _
4       been    be      AUX     PERF    Tense=Past|VerbForm=Part        5       aux:pass        _       _
5       sent    send    VERB    PASS    Tense=Past|VerbForm=Part|Voice=Pass     16      ccomp   _       _
```

So, finally, let us try to express the relation we want in plain English:

- A node with `'s` and lemma `be`...
- which is followed by an `AUX`...
- with no words other than `ADV` between them...
- and they are both children of the same parent

First we express that the immediate word after is any `AUX`, not just `be`, and indeed all of the matches for this are correct:

```
python3 stanza/server/semgrex.py  --input_file UD_English-LinES/en_lines-ud-train.conllu "{} > {word:/'s/;lemma:be}=has > {cpos:AUX}=other : {}=has . {}=other" --matches_only
```

To only allow `ADV` in between, we can first try this pattern:

```
"{} > {word:/'s/;lemma:be}=has > {cpos:AUX}=other : {}=has .. ({cpos:ADV} .. {}=other)"
```

There is a theoretical problem here, which is that this is *positive*
matches, and says nothing about any other words in between `has` and
`other`.  As it turns out, we are lucky in this particular treebank,
and there are no spurious sentences that match.  If there were, we
could eliminate them with a negative regex, such as

```
{}=has !.. ({cpos:/((?!ADV).*)/} .. {}=other)
```

Up until now, we have been doing everything with shell commands, so
one would either need to escape the shell symbols such as `!` or
simply use the `--semgrex_file` flag.

In the second sentence from our original results, we found:

```
# text = Rebecca's been to the Sputnik and she says it's terrific now.
```

Here, `'s` is a dependant of `been`.  Hypothetically, the sentence could be

```
# text = Rebecca's already been to the Sputnik and she says it's terrific now.
```

Since `'s` is a dependant of `been`, it is much easier to find:

```
python3 stanza/server/semgrex.py  --input_file UD_English-LinES/en_lines-ud-train.conllu "{word:/'s/;lemma:be}=has < {cpos:AUX}" --matches_only
```

This picks up one more sentence:

```
# text = He's just been to Denmark or somewhere because his mother died.
```

To summarize:

```
python3 stanza/server/semgrex.py  --input_file UD_English-LinES/en_lines-ud-train.conllu "{} > {word:/'s/;lemma:be}=has > {cpos:AUX}=other : {}=has . {}=other" --matches_only 
python3 stanza/server/semgrex.py  --input_file UD_English-LinES/en_lines-ud-train.conllu "{} > {word:/'s/;lemma:be}=has > {cpos:AUX}=other : {}=has .. ({cpos:ADV} .. {}=other)" --matches_only
python3 stanza/server/semgrex.py  --input_file UD_English-LinES/en_lines-ud-train.conllu "{word:/'s/;lemma:be}=has < {cpos:AUX}" --matches_only
```

and apply to each of train, dev, and test.

## Example Ssurgeon edit

There is also a Python interface to Ssurgeon, which uses Semgrex to search for patterns and then edits the matching trees based on an edit script.

The edit to change the lemma of a word is really quite simple:

```
editNode -node <name> -lemma <lemma>
```

so in this case, we do

```
python3 stanza/server/ssurgeon.py  --input_file UD_English-LinES/en_lines-ud-train.conllu --semgrex  "{word:/'s/;lemma:be}=has < {cpos:AUX}"  "editNode -node has -lemma have" --output_file en_lines-ud_train.conllu
```

We can do an entire directory at once with the `--input_dir` and `--output_dir` flags:

```
python3 stanza/server/ssurgeon.py  --input_dir UD_English-LinES --semgrex  "{word:/'s/;lemma:be}=has < {cpos:AUX}"  "editNode -node has -lemma have" --output_dir remapped
```

There is also the ability to provide an entire edit script, so that it is not necessary to run each of the three semgrex expressions listed above.  For example, this script:

```
{word:/'s/;lemma:be}=has < {cpos:AUX}
editNode -node has -lemma have

{} > {word:/'s/;lemma:be}=has > {cpos:AUX}=other : {}=has .. ({cpos:ADV} .. {}=other)
editNode -node has -lemma have

{} > {word:/'s/;lemma:be}=has > {cpos:AUX}=other : {}=has . {}=other
editNode -node has -lemma have
```

will edit all of the `'s` we found above at once:

```
python3 stanza/server/ssurgeon.py  --input_dir UD_English-LinES --edit_file has_edit.txt --output_dir remapped
```

## Combining MWT

There is a wide variety of Ssurgeon edits possible.  Another is
`combineMWT`, which marks some nodes as MWT.  For example, the
following edit script will capture many of the MWT and mark them:

```
{}=first . {word:/'s|n't|'ll|'ve/}=second
combineMWT -node first -node second
```

## More Information

More information on Semgrex patterns is available on its [javadoc page](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/semgraph/semgrex/SemgrexPattern.html)

More information on Ssurgeon is available on its [javadoc page](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/semgraph/semgrex/ssurgeon/Ssurgeon.html)

A writeup of Semgrex and Ssurgeon was published at [GURT 2023](https://aclanthology.org/2023.tlt-1.7/)