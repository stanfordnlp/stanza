"""Invokes the Java tokensregex on a document

This operates tokensregex on docs processed with stanza models.

https://nlp.stanford.edu/software/tokensregex.html

A minimal example is the main method of this module.
"""

import stanza

from stanza.protobuf import TokensRegexRequest, TokensRegexResponse
from stanza.server.java_protobuf_requests import send_request, add_token

def send_tokensregex_request(request):
    return send_request(request, TokensRegexResponse,
                        "edu.stanford.nlp.ling.tokensregex.ProcessTokensRegexRequest")

def process_doc(doc, *patterns):
    request = TokensRegexRequest()
    for pattern in patterns:
        request.pattern.append(pattern)

    request_doc = request.doc
    request_doc.text = doc.text
    num_tokens = 0
    for sent_idx, sentence in enumerate(doc.sentences):
        request_sentence = request_doc.sentence.add()
        request_sentence.tokenOffsetBegin = num_tokens
        request_sentence.tokenOffsetEnd = num_tokens + len(sentence.tokens)
        num_tokens = num_tokens + len(sentence.tokens)
        for token in sentence.tokens:
            for word in token.words:
                add_token(request_sentence.token, word, token)

    return send_tokensregex_request(request)

def main():
    #nlp = stanza.Pipeline('en',
    #                      processors='tokenize,pos,lemma,ner')
    nlp = stanza.Pipeline('en',
                          processors='tokenize')

    doc = nlp('Uro ruined modern.  Fortunately, Wotc banned him')
    print(process_doc(doc, "him", "ruined"))


if __name__ == '__main__':
    main()
