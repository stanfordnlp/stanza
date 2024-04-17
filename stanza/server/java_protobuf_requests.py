from collections import deque
import subprocess

from stanza.models.common.utils import misc_to_space_after
from stanza.models.constituency.parse_tree import Tree
from stanza.protobuf import DependencyGraph, FlattenedParseTree
from stanza.server.client import resolve_classpath

def send_request(request, response_type, java_main, classpath=None):
    """
    Use subprocess to run a Java protobuf processor on the given request

    Returns the protobuf response
    """
    classpath = resolve_classpath(classpath)
    if classpath is None:
        raise ValueError("Classpath is None,  Perhaps you need to set the $CLASSPATH or $CORENLP_HOME environment variable to point to a CoreNLP install.")
    pipe = subprocess.run(["java", "-cp", classpath, java_main],
                          input=request.SerializeToString(),
                          stdout=subprocess.PIPE,
                          check=True)
    response = response_type()
    response.ParseFromString(pipe.stdout)
    return response

def add_tree_nodes(proto_tree, tree, score):
    # add an open node
    node = proto_tree.nodes.add()
    node.openNode = True
    if score is not None:
        node.score = score

    # add the content of this node
    node = proto_tree.nodes.add()
    node.value = tree.label

    # add all children...
    # leaves get just one node
    # branches are called recursively
    for child in tree.children:
        if child.is_leaf():
            node = proto_tree.nodes.add()
            node.value = child.label
        else:
            add_tree_nodes(proto_tree, child, None)

    node = proto_tree.nodes.add()
    node.closeNode = True

def build_tree(tree, score):
    """
    Builds a FlattenedParseTree from CoreNLP.proto

    Populates the value field from tree.label and iterates through the
    children via tree.children.  Should work on any tree structure
    which follows that layout

    The score will be added to the top node (if it is not None)

    Operates by recursively calling add_tree_nodes
    """
    proto_tree = FlattenedParseTree()
    add_tree_nodes(proto_tree, tree, score)
    return proto_tree

def from_tree(proto_tree):
    """
    Convert a FlattenedParseTree back into a Tree

    returns Tree, score
      (score might be None if it is missing)
    """
    score = None
    stack = deque()
    for node in proto_tree.nodes:
        if node.HasField("score") and score is None:
            score = node.score

        if node.openNode:
            if len(stack) > 0 and isinstance(stack[-1], FlattenedParseTree.Node) and stack[-1].openNode:
                raise ValueError("Got a proto with no label on a node: {}".format(proto_tree))
            stack.append(node)
            continue
        if not node.closeNode:
            child = Tree(label=node.value)
            # TODO: do something with the score
            stack.append(child)
            continue

        # must be a close operation...
        if len(stack) <= 1:
            raise ValueError("Got a proto with too many close operations: {}".format(proto_tree))
        # on a close operation, pop until we hit the open
        # then turn everything in that span into a new node
        children = []
        nextNode = stack.pop()
        while not isinstance(nextNode, FlattenedParseTree.Node):
            children.append(nextNode)
            nextNode = stack.pop()
        if len(children) == 0:
            raise ValueError("Got a proto with an open immediately followed by a close: {}".format(proto_tree))
        children.reverse()
        label = children[0]
        children = children[1:]
        subtree = Tree(label=label.label, children=children)
        stack.append(subtree)

    if len(stack) > 1:
        raise ValueError("Got a proto which does not close all of the nodes: {}".format(proto_tree))
    tree = stack.pop()
    if not isinstance(tree, Tree):
        raise ValueError("Got a proto which was just one Open operation: {}".format(proto_tree))
    return tree, score

def add_token(token_list, word, token):
    """
    Add a token to a proto request.

    CoreNLP tokens have components of both word and token from stanza.

    We pass along "after" but not "before"
    """
    if token is None and isinstance(word.id, int):
        raise AssertionError("Only expected word w/o token for 'extra' words")

    query_token = token_list.add()
    query_token.word = word.text
    query_token.value = word.text
    if word.lemma is not None:
        query_token.lemma = word.lemma
    if word.xpos is not None:
        query_token.pos = word.xpos
    if word.upos is not None:
        query_token.coarseTag = word.upos
    if word.feats and word.feats != "_":
        for feature in word.feats.split("|"):
            key, value = feature.split("=", maxsplit=1)
            query_token.conllUFeatures.key.append(key)
            query_token.conllUFeatures.value.append(value)
    if token is not None:
        if token.ner is not None:
            query_token.ner = token.ner
        if token is not None and len(token.id) > 1:
            query_token.mwtText = token.text
            query_token.isMWT = True
            query_token.isFirstMWT = token.id[0] == word.id
        if token.id[-1] != word.id:
            # if we are not the last word of an MWT token
            # we are absolutely not followed by space
            pass
        else:
            query_token.after = token.spaces_after

        query_token.index = word.id
    else:
        # presumably empty words won't really be written this way,
        # but we can still keep track of it
        query_token.after = misc_to_space_after(word.misc)

        query_token.index = word.id[0]
        query_token.emptyIndex = word.id[1]

    if word.misc and word.misc != "_":
        query_token.conllUMisc = word.misc
    if token is not None and token.misc and token.misc != "_":
        query_token.mwtMisc = token.misc

def add_sentence(request_sentences, sentence, num_tokens):
    """
    Add the tokens for this stanza sentence to a list of protobuf sentences
    """
    request_sentence = request_sentences.add()
    request_sentence.tokenOffsetBegin = num_tokens
    request_sentence.tokenOffsetEnd = num_tokens + sum(len(token.words) for token in sentence.tokens)
    for token in sentence.tokens:
        for word in token.words:
            add_token(request_sentence.token, word, token)
    return request_sentence

def add_word_to_graph(graph, word, sent_idx, word_idx):
    """
    Add a node and possibly an edge for a word in a basic dependency graph.
    """
    node = graph.node.add()
    node.sentenceIndex = sent_idx+1
    node.index = word_idx+1

    if word.head != 0 and word.head is not None:
        edge = graph.edge.add()
        edge.source = word.head
        edge.target = word_idx+1
        if word.deprel is not None:
            edge.dep = word.deprel
        else:
            # the receiving side doesn't like null as a dependency
            edge.dep = "_"

def convert_networkx_graph(graph_proto, sentence, sent_idx):
    """
    Turns a networkx graph into a DependencyGraph from the proto file
    """
    for token in sentence.tokens:
        for word in token.words:
            add_token(graph_proto.token, word, token)
    for word in sentence.empty_words:
        add_token(graph_proto.token, word, None)

    dependencies = sentence._enhanced_dependencies
    for target in dependencies:
        if target == 0:
            # don't need to send the explicit root
            continue
        for source in dependencies.predecessors(target):
            if source == 0:
                # unlike with basic, we need to send over the roots,
                # as the enhanced can have loops
                graph_proto.rootNode.append(len(graph_proto.node))
                continue
            for deprel in dependencies.get_edge_data(source, target):
                edge = graph_proto.edge.add()
                if isinstance(source, int):
                    edge.source = source
                else:
                    edge.source = source[0]
                    if source[1] != 0:
                        edge.sourceEmpty = source[1]
                if isinstance(target, int):
                    edge.target = target
                else:
                    edge.target = target[0]
                    if target[1] != 0:
                        edge.targetEmpty = target[1]
                edge.dep = deprel
        node = graph_proto.node.add()
        node.sentenceIndex = sent_idx + 1
        # the nodes in the networkx graph are indexed from 1, not counting the root
        if isinstance(target, int):
            node.index = target
        else:
            node.index = target[0]
            if target[1] != 0:
                node.emptyIndex = target[1]
    return graph_proto

def features_to_string(features):
    if not features:
        return None
    if len(features.key) == 0:
        return None
    return "|".join("%s=%s" % (key, value) for key, value in zip(features.key, features.value))

def misc_space_pieces(misc):
    """
    Return only the space-related misc pieces
    """
    if misc is None or misc == "" or misc == "_":
        return misc
    pieces = misc.split("|")
    pieces = [x for x in pieces if x.split("=", maxsplit=1)[0] in ("SpaceAfter", "SpacesAfter", "SpacesBefore")]
    if len(pieces) > 0:
        return "|".join(pieces)
    return None

def remove_space_misc(misc):
    """
    Remove any pieces from misc which are space-related
    """
    if misc is None or misc == "" or misc == "_":
        return misc
    pieces = misc.split("|")
    pieces = [x for x in pieces if x.split("=", maxsplit=1)[0] not in ("SpaceAfter", "SpacesAfter", "SpacesBefore")]
    if len(pieces) > 0:
        return "|".join(pieces)
    return None

def substitute_space_misc(misc, space_misc):
    space_misc_pieces = space_misc.split("|") if space_misc else []
    space_misc_after = None
    space_misc_before = None
    for piece in space_misc_pieces:
        if piece.startswith("SpaceBefore"):
            space_misc_before = piece
        elif piece.startswith("SpaceAfter") or piece.startswith("SpacesAfter"):
            space_misc_after = piece
        else:
            raise AssertionError("An unknown piece wound up in the misc space fields: %s" % piece)

    pieces = misc.split("|")
    new_pieces = []
    for piece in pieces:
        if piece.startswith("SpaceBefore"):
            if space_misc_before:
                new_pieces.append(space_misc_before)
                space_misc_before = None
        elif piece.startswith("SpaceAfter") or piece.startswith("SpacesAfter"):
            if space_misc_after:
                new_pieces.append(space_misc_after)
                space_misc_after = None
        else:
            new_pieces.append(piece)
    if space_misc_after:
        new_pieces.append(space_misc_after)
    if space_misc_before:
        new_pieces.append(space_misc_before)
    if len(new_pieces) == 0:
        return None
    return "|".join(new_pieces)

class JavaProtobufContext(object):
    """
    A generic context for sending requests to a java program using protobufs in a subprocess
    """
    def __init__(self, classpath, build_response, java_main, extra_args=None):
        self.classpath = resolve_classpath(classpath)
        self.build_response = build_response
        self.java_main = java_main

        if extra_args is None:
            extra_args = []
        self.extra_args = extra_args
        self.pipe = None

    def open_pipe(self):
        self.pipe = subprocess.Popen(["java", "-cp", self.classpath, self.java_main, "-multiple"] + self.extra_args,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)

    def close_pipe(self):
        if self.pipe.poll() is None:
            self.pipe.stdin.write((0).to_bytes(4, 'big'))
            self.pipe.stdin.flush()
            self.pipe = None

    def __enter__(self):
        self.open_pipe()
        return self

    def __exit__(self, type, value, traceback):
        self.close_pipe()

    def process_request(self, request):
        if self.pipe is None:
            raise RuntimeError("Pipe to java process is not open or was closed")

        text = request.SerializeToString()
        self.pipe.stdin.write(len(text).to_bytes(4, 'big'))
        self.pipe.stdin.write(text)
        self.pipe.stdin.flush()
        response_length = self.pipe.stdout.read(4)
        if len(response_length) < 4:
            raise BrokenPipeError("Could not communicate with java process!")
        response_length = int.from_bytes(response_length, "big")
        response_text = self.pipe.stdout.read(response_length)
        response = self.build_response()
        response.ParseFromString(response_text)
        return response

