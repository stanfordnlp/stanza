"""Invokes the Java ssurgeon on a document

"ssurgeon" sends text to Java CoreNLP for processing with a ssurgeon
(Semantic graph SURGEON) query

The main program in this file gives a very short intro to how to use it.
"""


import argparse

from stanza.protobuf import SsurgeonRequest, SsurgeonResponse
from stanza.server.java_protobuf_requests import send_request, add_token, add_word_to_graph, JavaProtobufContext
from stanza.utils.conll import CoNLL

SSURGEON_JAVA = "edu.stanford.nlp.semgraph.semgrex.ssurgeon.ProcessSsurgeonRequest"

class SsurgeonEdit:
    def __init__(self, semgrex_pattern, ssurgeon_edits, ssurgeon_id=None, notes=None, language="UniversalEnglish"):
        # not a named tuple so we can have defaults without requiring a python upgrade
        self.semgrex_pattern = semgrex_pattern
        self.ssurgeon_edits = ssurgeon_edits
        self.ssurgeon_id = ssurgeon_id
        self.notes = notes
        self.language = language

def send_ssurgeon_request(request):
    return send_request(request, SsurgeonResponse, SSURGEON_JAVA)

def build_request(doc, ssurgeon_edits):
    request = SsurgeonRequest()

    for ssurgeon in ssurgeon_edits:
        ssurgeon_proto = request.ssurgeon.add()
        ssurgeon_proto.semgrex = ssurgeon.semgrex_pattern
        for operation in ssurgeon.ssurgeon_edits:
            ssurgeon_proto.operation.append(operation)
        if ssurgeon.ssurgeon_id is not None:
            ssurgeon_proto.id = ssurgeon.ssurgeon_id
        if ssurgeon.notes is not None:
            ssurgeon_proto.notes = ssurgeon.notes
        if ssurgeon.language is not None:
            ssurgeon_proto.language = ssurgeon.language

    for sent_idx, sentence in enumerate(doc.sentences):
        graph = request.graph.add()
        word_idx = 0
        for token in sentence.tokens:
            for word in token.words:
                add_token(graph.token, word, token)
                add_word_to_graph(graph, word, sent_idx, word_idx)

                word_idx = word_idx + 1

    return request

def build_request_one_operation(doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id=None, notes=None):
    ssurgeon_edit = SsurgeonEdit(semgrex_pattern, ssurgeon_edits, ssurgeon_id, notes)
    return build_request(doc, [ssurgeon_edit])

def process_doc(doc, ssurgeon_edits):
    """
    Returns the result of processing the given semgrex expression and ssurgeon edits on the stanza doc.

    Currently the return is a SsurgeonResponse from CoreNLP.proto
    """
    request = build_request(doc, ssurgeon_edits)

    return send_ssurgeon_request(request)

def process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id=None, notes=None):
    request = build_request_one_operation(doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id, notes)

    return send_ssurgeon_request(request)

class Ssurgeon(JavaProtobufContext):
    """
    Ssurgeon context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, classpath=None):
        super(Ssurgeon, self).__init__(classpath, SsurgeonResponse, SSURGEON_JAVA)

    def process(self, doc, ssurgeon_edits):
        """
        Apply each of the ssurgeon patterns to each of the dependency trees in doc
        """
        request = build_request(doc, ssurgeon_edits)
        return self.process_request(request)

    def process_one_operation(self, doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id=None, notes=None):
        """
        Convenience method - build one operation, then apply it
        """
        request = build_request_one_operation(doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id, notes)
        return self.process_request(request)

SAMPLE_DOC = """
# sent_id = 271
# text = Hers is easy to clean.
# previous = What did the dealer like about Alex's car?
# comment = extraction/raising via "tough extraction" and clausal subject
1	Hers	hers	PRON	PRP	Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs	3	nsubj	_	_
2	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	cop	_	_
3	easy	easy	ADJ	JJ	Degree=Pos	0	root	_	_
4	to	to	PART	TO	_	5	mark	_	_
5	clean	clean	VERB	VB	VerbForm=Inf	3	csubj	_	SpaceAfter=No
6	.	.	PUNCT	.	_	5	punct	_	_
"""

def main():
    # The default semgrex detects sentences in the UD_English-Pronouns dataset which have both nsubj and csubj on the same word.
    # The default ssurgeon transforms the unwanted csubj to advcl:relcl.
    # See https://github.com/UniversalDependencies/docs/issues/923
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=None, help="Input file to process (otherwise will process a sample text)")
    parser.add_argument('--semgrex', type=str, default="{}=source >nsubj {} >csubj=bad {}", help="Semgrex to apply to the text.  A default detects words which have both an nsubj and a csubj")
    parser.add_argument('ssurgeon', type=str, nargs="*", help="Ssurgeon edits to apply based on the Semgrex.  Can have multiple edits in a row.  A default exists to transform csubj into advcl:relcl")
    args = parser.parse_args()

    if len(args.ssurgeon) == 0:
        args.ssurgeon = ["relabelNamedEdge -edge bad -reln advcl:relcl"]

    if args.input_file:
        doc = CoNLL.conll2doc(input_file=args.input_file)
    else:
        doc = CoNLL.conll2doc(input_str=SAMPLE_DOC)

    print("{:C}".format(doc))
    print(process_doc_one_operation(doc, args.semgrex, args.ssurgeon))

if __name__ == '__main__':
    main()
