"""
Invokes the Java semgrex on a document
"""

import subprocess

import stanza
from stanza.protobuf import SemgrexRequest, SemgrexResponse

def send_request(request):
    """
    Use subprocess to run the Semgrex processor on the given request

    Returns the protobuf response
    """
    pipe = subprocess.run("java edu.stanford.nlp.semgraph.semgrex.ProcessSemgrexRequest".split(),
                          input=request.SerializeToString(),
                          stdout=subprocess.PIPE)
    response = SemgrexResponse()
    response.ParseFromString(pipe.stdout)
    return response


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

    return send_request(request)

def main():
    nlp = stanza.Pipeline('en',
                          processors='tokenize,pos,lemma,depparse')

    doc = nlp('Unban Mox Opal! Unban Mox Opal!')
    #print(doc.sentences[0].dependencies)
    print(doc)
    print(process_doc(doc, "{}=source >obj=zzz {}=target"))

if __name__ == '__main__':
    main()
