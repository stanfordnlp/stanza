


import stanza
from stanza.protobuf import EvaluateParserRequest, EvaluateParserResponse
from stanza.server.java_protobuf_requests import send_request, build_tree, JavaProtobufContext


EVALUATE_JAVA = "edu.stanford.nlp.parser.metrics.EvaluateExternalParser"

def build_request(treebank):
    """
    treebank should be a list of pairs:  [gold, predictions]
      each predictions is a list of pairs (prediction, score)
    Note that for now, only one tree is measured, but this may be extensible in the future
    Trees should be in the form of a Tree from parse_tree.py
    """
    request = EvaluateParserRequest()
    for gold, predictions in treebank:
        parse_result = request.treebank.add()
        parse_result.gold.CopyFrom(build_tree(gold, None))
        for prediction, score in predictions:
            parse_result.predicted.append(build_tree(prediction, score))

    return request


class EvaluateParser(JavaProtobufContext):
    """
    Parser evaluation context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, classpath=None):
        super(EvaluateParser, self).__init__(classpath, EvaluateParserResponse, EVALUATE_JAVA)

    def process(self, treebank):
        request = build_request(treebank)
        return self.process_request(request)

