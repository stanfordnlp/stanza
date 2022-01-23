"""Invokes the Java tsurgeon on a list of trees

Included with CoreNLP is a mechanism for modifying trees based on
existing patterns within a tree.  The patterns are found using tregex:

https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/trees/tregex/TregexPattern.html

The modifications are then performed using tsurgeon:

https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/trees/tregex/tsurgeon/Tsurgeon.html

This module accepts Tree objects as produced by the conparser and
returns the modified trees that result from one or more tsurgeon
operations.
"""

from stanza.models.constituency import tree_reader
from stanza.models.constituency.parse_tree import Tree
from stanza.protobuf import TsurgeonRequest, TsurgeonResponse
from stanza.server.java_protobuf_requests import send_request, build_tree, from_tree, JavaProtobufContext

TSURGEON_JAVA = "edu.stanford.nlp.trees.tregex.tsurgeon.ProcessTsurgeonRequest"

def send_tsurgeon_request(request):
    return send_request(request, TsurgeonResponse, TSURGEON_JAVA)


def build_request(trees, operations):
    """
    Build the TsurgeonRequest object

    trees: a list of trees
    operations: a list of (tregex, tsurgeon, tsurgeon, ...)
    """
    if isinstance(trees, Tree):
        trees = (trees,)

    request = TsurgeonRequest()
    for tree in trees:
        request.trees.append(build_tree(tree, 0.0))
    if all(isinstance(x, str) for x in operations):
        operations = (operations,)
    for operation in operations:
        if len(operation) == 1:
            raise ValueError("Expected [tregex, tsurgeon, ...] but just got a tregex")
        operation_request = request.operations.add()
        operation_request.tregex = operation[0]
        for tsurgeon in operation[1:]:
            operation_request.tsurgeon.append(tsurgeon)
    return request


def process_trees(trees, *operations):
    """
    Returns the result of processing the given tsurgeon operations on the given trees

    Returns a list of modified trees, eg, the result is already processed
    """
    request = build_request(trees, operations)
    result = send_tsurgeon_request(request)

    return [from_tree(t)[0] for t in result.trees]


class Tsurgeon(JavaProtobufContext):
    """
    Tsurgeon context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, classpath=None):
        super(Tsurgeon, self).__init__(classpath, TsurgeonResponse, TSURGEON_JAVA)

    def process(self, trees, *operations):
        request = build_request(trees, operations)
        result = self.process_request(request)
        return [from_tree(t)[0] for t in result.trees]


def main():
    """
    A small demonstration of a tsurgeon operation
    """
    text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = tree_reader.read_trees(text)

    tregex = "WP=wp"
    tsurgeon = "relabel wp WWWPPP"

    result = process_trees(trees, (tregex, tsurgeon))
    print(result)

if __name__ == '__main__':
    main()
