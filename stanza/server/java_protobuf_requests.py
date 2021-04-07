import subprocess

from stanza.server.client import resolve_classpath

def send_request(request, response_type, java_main, classpath=None):
    """
    Use subprocess to run a Java protobuf processor on the given request

    Returns the protobuf response
    """
    pipe = subprocess.run(["java", "-cp", resolve_classpath(classpath), java_main],
                          input=request.SerializeToString(),
                          stdout=subprocess.PIPE,
                          check=True)
    response = response_type()
    response.ParseFromString(pipe.stdout)
    return response

def add_token(token_list, word, token):
    """
    Add a token to a proto request.

    CoreNLP tokens have components of both word and token from stanza.
    """
    query_token = token_list.add()
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

    if word.head != 0:
        edge = graph.edge.add()
        edge.source = word.head
        edge.target = word_idx+1
        edge.dep = word.deprel

class JavaProtobufContext(object):
    """
    A generic context for sending requests to a java program using protobufs in a subprocess
    """
    def __init__(self, classpath, build_response, java_main):
        self.classpath = resolve_classpath(classpath)
        self.build_response = build_response
        self.java_main = java_main


    def __enter__(self):
        self.pipe = subprocess.Popen(["java", "-cp", self.classpath, self.java_main, "-multiple"],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        return self

    def __exit__(self, type, value, traceback):
        if self.pipe.poll() is not None:
            self.pipe.stdin.write((0).to_bytes(4, 'big'))
            self.pipe.stdin.flush()

    def process_request(self, request):
        text = request.SerializeToString()
        self.pipe.stdin.write(len(text).to_bytes(4, 'big'))
        self.pipe.stdin.write(text)
        self.pipe.stdin.flush()
        response_length = self.pipe.stdout.read(4)
        if len(response_length) < 4:
            raise RuntimeError("Could not communicate with java process!")
        response_length = int.from_bytes(response_length, "big")
        response_text = self.pipe.stdout.read(response_length)
        response = self.build_response()
        response.ParseFromString(response_text)
        return response

