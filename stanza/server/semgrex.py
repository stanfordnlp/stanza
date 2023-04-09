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
pattern, as that would serialize the document each time.
Included here is a context manager which allows for keeping the same
java process open for multiple requests.  This saves on the subprocess
launching time.  It is still important not to wastefully serialize the
same document over and over, though.
"""

import argparse
import copy

import stanza
from stanza.protobuf import SemgrexRequest, SemgrexResponse
from stanza.server.java_protobuf_requests import send_request, add_token, add_word_to_graph, JavaProtobufContext
from stanza.utils.conll import CoNLL

SEMGREX_JAVA = "edu.stanford.nlp.semgraph.semgrex.ProcessSemgrexRequest"

def send_semgrex_request(request):
    return send_request(request, SemgrexResponse, SEMGREX_JAVA)

def build_request(doc, semgrex_patterns):
    request = SemgrexRequest()
    for semgrex in semgrex_patterns:
        request.semgrex.append(semgrex)

    for sent_idx, sentence in enumerate(doc.sentences):
        query = request.query.add()
        word_idx = 0
        for token in sentence.tokens:
            for word in token.words:
                add_token(query.token, word, token)
                add_word_to_graph(query.graph, word, sent_idx, word_idx)

                word_idx = word_idx + 1

    return request

def process_doc(doc, *semgrex_patterns):
    """
    Returns the result of processing the given semgrex expression on the stanza doc.

    Currently the return is a SemgrexResponse from CoreNLP.proto
    """
    request = build_request(doc, semgrex_patterns)

    return send_semgrex_request(request)

class Semgrex(JavaProtobufContext):
    """
    Semgrex context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, classpath=None):
        super(Semgrex, self).__init__(classpath, SemgrexResponse, SEMGREX_JAVA)

    def process(self, doc, *semgrex_patterns):
        """
        Apply each of the semgrex patterns to each of the dependency trees in doc
        """
        request = build_request(doc, semgrex_patterns)
        return self.process_request(request)

def annotate_doc(doc, semgrex_result, semgrex_patterns, matches_only):
    """
    Put comments on the sentences which describe the matching semgrex patterns
    """
    doc = copy.deepcopy(doc)
    if isinstance(semgrex_patterns, str):
        semgrex_patterns = [semgrex_patterns]
    matching_sentences = []
    for sentence, graph_result in zip(doc.sentences, semgrex_result.result):
        sentence_matched = False
        for semgrex_pattern, pattern_result in zip(semgrex_patterns, graph_result.result):
            semgrex_pattern = semgrex_pattern.replace("\n", " ")
            if len(pattern_result.match) == 0:
                sentence.add_comment("# semgrex pattern |%s| did not match!" % semgrex_pattern)
            else:
                sentence_matched = True
                for match in pattern_result.match:
                    match_word = "%d:%s" % (match.matchIndex, sentence.words[match.matchIndex-1].text)
                    if len(match.node) == 0:
                        node_matches = ""
                    else:
                        node_matches = ["%s=%d:%s" % (node.name, node.matchIndex, sentence.words[node.matchIndex-1].text)
                                        for node in match.node]
                        node_matches = "  " + " ".join(node_matches)
                    sentence.add_comment("# semgrex pattern |%s| matched at %s%s" % (semgrex_pattern, match_word, node_matches))
        if sentence_matched:
            matching_sentences.append(sentence)
    if matches_only:
        doc.sentences = matching_sentences
    return doc


def main():
    """
    Runs a toy example, or can run a given semgrex expression on the given input file.

    For example:
    python3 -m stanza.server.semgrex --input_file demo/semgrex_sample.conllu

    --matches_only to only print sentences that match the semgrex pattern
    --no_print_input to not print the input
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=None, help="Input file to process (otherwise will process a sample text)")
    parser.add_argument('semgrex', type=str, nargs="*", default=["{}=source >obj=zzz {}=target"], help="Semgrex to apply to the text.  The default looks for sentences with objects")
    parser.add_argument('--semgrex_file', type=str, default=None, help="File to read semgrex patterns from - relevant in case the pattern you want to use doesn't work well on the command line, for example")
    parser.add_argument('--print_input', dest='print_input', action='store_true', default=False, help="Print the input alongside the output - gets kind of noisy")
    parser.add_argument('--no_print_input', dest='print_input', action='store_false', help="Don't print the input alongside the output - gets kind of noisy")
    parser.add_argument('--matches_only', action='store_true', default=False, help="Only print the matching sentences")
    args = parser.parse_args()

    if args.semgrex_file:
        with open(args.semgrex_file) as fin:
            args.semgrex = [x.strip() for x in fin.readlines() if x.strip()]

    if args.input_file:
        doc = CoNLL.conll2doc(input_file=args.input_file)
    else:
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')
        doc = nlp('Uro ruined modern.  Fortunately, Wotc banned him.')

    if args.print_input:
        print("{:C}".format(doc))
        print()
        print("-" * 75)
        print()
    semgrex_result = process_doc(doc, *args.semgrex)
    doc = annotate_doc(doc, semgrex_result, args.semgrex, args.matches_only)
    print("{:C}".format(doc))

if __name__ == '__main__':
    main()
