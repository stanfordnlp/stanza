"""Invokes the Java semgrex on a document

The server client has a method "semgrex" which sends text to Java
CoreNLP for processing with a semgrex (SEMantic GRaph regEX) query:

https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/semgraph/semgrex/SemgrexPattern.html

However, this operates on text using the CoreNLP tools, which means
the dependency graphs may not align with stanza's depparse module, and
this also limits the languages for which it can be used.  This module
allows for running semgrex commands on the graphs produced by
depparse.

To use, first process text into a doc using stanza.Pipeline

Next, pass the processed doc and a list of semgrex patterns to
process_doc in this module.  It will run the java semgrex module as a
subprocess and return the result in the form of a SemgrexResponse,
whose description is in the proto file included with stanza.

A minimal example is the main method of this module.

Note that launching the subprocess is potentially quite expensive
relative to the search if used many times on small documents.  Ideally
larger texts would be processed, and all of the desired semgrex
patterns would be run at once.  The worst thing to do would be to call
this multiple times on a large document, one invocation per semgrex
pattern, as that would serialize the document each time.  There are of
course multiple ways of making this more efficient, such as including
it as a separate call in the server or keeping the subprocess alive
for multiple queries, but we didn't do any of those.  We do, however,
accept pull requests...
"""

import subprocess

import stanza
from stanza.protobuf import SemgrexRequest, SemgrexResponse
from stanza.server.client import resolve_classpath

def send_request(request, response_type, java_main):
    """
    Use subprocess to run the Semgrex processor on the given request

    Returns the protobuf response
    """
    pipe = subprocess.run(["java", "-cp", resolve_classpath(), java_main],
                          input=request.SerializeToString(),
                          stdout=subprocess.PIPE)
    response = response_type()
    response.ParseFromString(pipe.stdout)
    return response

def send_semgrex_request(request):
    return send_request(request, SemgrexResponse,
                        "edu.stanford.nlp.semgraph.semgrex.ProcessSemgrexRequest")

def process_doc(doc, *semgrex_patterns):
    """
    Returns the result of processing the given semgrex expression on the stanza doc.

    Currently the return is a SemgrexResponse from CoreNLP.proto
    """
    request = SemgrexRequest()
    for semgrex in semgrex_patterns:
        request.semgrex.append(semgrex)

    for sent_idx, sentence in enumerate(doc.sentences):
        query = request.query.add()
        word_idx = 0
        for token in sentence.tokens:
            for word in token.words:
                query_token = query.token.add()
                query_token.word = word.text
                query_token.value = word.text
                if word.lemma is not None:
                    query_token.lemma = word.lemma
                if word.xpos is not None:
                    query_token.pos = word.xpos
                if word.upos is not None:
                    query_token.coarseTag = word.upos
                if token.ner is not None:
                    query_token.ner = token.ner

                node = query.graph.node.add()
                node.sentenceIndex = sent_idx+1
                node.index = word_idx+1

                if word.head != 0:
                    edge = query.graph.edge.add()
                    edge.source = word.head
                    edge.target = word_idx+1
                    edge.dep = word.deprel

                word_idx = word_idx + 1

    return send_semgrex_request(request)

def main():
    nlp = stanza.Pipeline('en',
                          processors='tokenize,pos,lemma,depparse')

    doc = nlp('Uro ruined modern.  Fortunately, Wotc banned him.')
    #print(doc.sentences[0].dependencies)
    print(doc)
    print(process_doc(doc, "{}=source >obj=zzz {}=target"))

if __name__ == '__main__':
    main()
