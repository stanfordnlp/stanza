"""
A converter from constituency trees to dependency trees using CoreNLP's UniversalEnglish converter.

ONLY works on English.
"""

import stanza
from stanza.protobuf import DependencyConverterRequest, DependencyConverterResponse
from stanza.server.java_protobuf_requests import send_request, build_tree, JavaProtobufContext

CONVERTER_JAVA = "edu.stanford.nlp.trees.ProcessDependencyConverterRequest"

def send_converter_request(request, classpath=None):
    return send_request(request, DependencyConverterResponse, CONVERTER_JAVA, classpath=classpath)

def build_request(doc):
    """
    Request format is simple: one tree per sentence in the document
    """
    request = DependencyConverterRequest()
    for sentence in doc.sentences:
        request.trees.append(build_tree(sentence.constituency, None))
    return request

def process_doc(doc, classpath=None):
    """
    Convert the constituency trees in the document,
    then attach the resulting dependencies to the sentences
    """
    request = build_request(doc)
    response = send_converter_request(request, classpath=classpath)
    attach_dependencies(doc, response)

def attach_dependencies(doc, response):
    if len(doc.sentences) != len(response.conversions):
        raise ValueError("Sent %d sentences but got back %d conversions" % (len(doc.sentences), len(response.conversions)))
    for sent_idx, (sentence, conversion) in enumerate(zip(doc.sentences, response.conversions)):
        graph = conversion.graph

        # The deterministic conversion should have an equal number of words and one fewer edge
        # ... the root is represented by a word with no parent
        if len(sentence.words) != len(graph.node):
            raise ValueError("Sentence %d of the conversion should have %d words but got back %d nodes in the graph" % (sent_idx, len(sentence.words), len(graph.node)))        
        if len(sentence.words) != len(graph.edge) + 1:
            raise ValueError("Sentence %d of the conversion should have %d edges (one per word, plus the root) but got back %d edges in the graph" % (sent_idx, len(sentence.words) - 1, len(graph.edge)))

        expected_nodes = set(range(1, len(sentence.words) + 1))
        targets = set()
        for edge in graph.edge:
            if edge.target in targets:
                raise ValueError("Found two parents of %d in sentence %d" % (edge.target, sent_idx))
            targets.add(edge.target)
            # -1 since the words are 0 indexed in the sentence,
            # but we count dependencies from 1
            sentence.words[edge.target-1].head = edge.source
            sentence.words[edge.target-1].deprel = edge.dep
        roots = expected_nodes - targets
        assert len(roots) == 1
        for root in roots:
            sentence.words[root-1].head = 0
            sentence.words[root-1].deprel = "root"
        sentence.build_dependencies()


class DependencyConverter(JavaProtobufContext):
    """
    Context window for the dependency converter

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, classpath=None):
        super(DependencyConverter, self).__init__(classpath, DependencyConverterResponse, CONVERTER_JAVA)

    def process(self, doc):
        """
        Converts a constituency tree to dependency trees for each of the sentences in the document
        """
        request = build_request(doc)
        response = self.process_request(request)
        attach_dependencies(doc, response)
        return doc

def main():
    nlp = stanza.Pipeline('en',
                          processors='tokenize,pos,constituency')

    doc = nlp('I like blue antennae.')
    print("{:C}".format(doc))
    process_doc(doc, classpath="$CLASSPATH")
    print("{:C}".format(doc))

    doc = nlp('And I cannot lie.')
    print("{:C}".format(doc))
    with DependencyConverter(classpath="$CLASSPATH") as converter:
        converter.process(doc)
        print("{:C}".format(doc))


if __name__ == '__main__':
    main()
