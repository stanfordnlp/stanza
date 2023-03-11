"""Invokes the Java ssurgeon on a document

"ssurgeon" sends text to Java CoreNLP for processing with a ssurgeon
(Semantic graph SURGEON) query

The main program in this file gives a very short intro to how to use it.
"""


import argparse
import copy

from stanza.protobuf import SsurgeonRequest, SsurgeonResponse
from stanza.server.java_protobuf_requests import send_request, add_token, add_word_to_graph, JavaProtobufContext, features_to_string
from stanza.utils.conll import CoNLL

from stanza.models.common.doc import ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, START_CHAR, END_CHAR, NER, Word, Token, Sentence

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

def convert_response_to_doc(doc, semgrex_response):
    doc = copy.deepcopy(doc)
    for sent_idx, (sentence, ssurgeon_result) in enumerate(zip(doc.sentences, semgrex_response.result)):
        if not ssurgeon_result.changed:
            continue

        ssurgeon_graph = ssurgeon_result.graph
        if len(ssurgeon_graph.token) == len(sentence.words) and all(x.word == y.text for x, y in zip(ssurgeon_graph.token, sentence.words)):
            # Word texts are unchanged.  Need to copy various attributes, plus the dependency links
            # TODO: pass back & forth the MWT.  the UD_English-Pronouns dataset can use that!
            #   for example, each usage of 's should attach to the previous
            #   possessive - dealer's
            #   it's
            #   isn't
            #   aint (no break)
            #   to be: car's
            #   'll as in hers'll, his'll, etc
            for graph_word, sentence_word in zip(ssurgeon_graph.token, sentence.words):
                sentence_word.lemma = graph_word.lemma
                sentence_word.upos = graph_word.coarseTag
                sentence_word.xpos = graph_word.pos
                sentence_word.head = None
                sentence_word.deprel = None
                sentence_word.deps = None
                sentence_word.feats = features_to_string(graph_word.conllUFeatures)
            for root in ssurgeon_graph.root:
                sentence.words[root-1].head = 0
                sentence.words[root-1].deprel = "root"
            for edge in ssurgeon_graph.edge:
                # can't do anything about the extra dependencies for now
                # TODO: put them all in .deps
                if edge.isExtra:
                    continue
                sentence.words[edge.target-1].head = edge.source
                sentence.words[edge.target-1].deprel = edge.dep
        else:
            # TODO: this will lose all the MWT
            #   There is probably a way to convey that to Ssurgeon and back
            # TODO: make that happen for the Pronouns dataset!
            tokens = []
            for graph_node, graph_word in zip(ssurgeon_graph.node, ssurgeon_graph.token):
                if graph_node.copyAnnotation:
                    continue
                word_entry = {
                    ID: graph_node.index,
                    TEXT: graph_word.word,
                    LEMMA: graph_word.lemma,
                    UPOS: graph_word.coarseTag,
                    XPOS: graph_word.pos,
                    FEATS: features_to_string(graph_word.conllUFeatures),
                    DEPS: None,
                    NER: graph_word.ner,
                    START_CHAR: None,   # TODO: fix this?  one problem is the text positions
                    END_CHAR: None,     #   might change across all of the sentences
                }
                if not graph_word.after:
                    word_entry[MISC] = "SpaceAfter=No"
                tokens.append(word_entry)
            tokens.sort(key=lambda x: x[ID])
            for root in ssurgeon_graph.root:
                tokens[root-1][HEAD] = 0
                tokens[root-1][DEPREL] = "root"
            for edge in ssurgeon_graph.edge:
                # can't do anything about the extra dependencies for now
                # TODO: put them all in .deps
                if edge.isExtra:
                    continue
                tokens[edge.target-1][HEAD] = edge.source
                tokens[edge.target-1][DEPREL] = edge.dep
            old_comments = list(sentence.comments)
            sentence = Sentence(tokens, doc)

            word_text = [word.text if (word_idx == len(sentence.words) - 1 or (word.misc and "SpaceAfter=No" in word.misc)) else word.text + " "
                         for word_idx, word in enumerate(sentence.words)]
            sentence_text = "".join(word_text)

            for comment in old_comments:
                if comment.startswith("# text"):
                    sentence.add_comment("# text = " + sentence_text)
                else:
                    sentence.add_comment(comment)
            
            doc.sentences[sent_idx] = sentence

        sentence.rebuild_dependencies()
    return doc

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
    # The default ssurgeon transforms the unwanted csubj to advcl
    # See https://github.com/UniversalDependencies/docs/issues/923
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=None, help="Input file to process (otherwise will process a sample text)")
    parser.add_argument('--semgrex', type=str, default="{}=source >nsubj {} >csubj=bad {}", help="Semgrex to apply to the text.  A default detects words which have both an nsubj and a csubj")
    parser.add_argument('ssurgeon', type=str, nargs="*", help="Ssurgeon edits to apply based on the Semgrex.  Can have multiple edits in a row.  A default exists to transform csubj into advcl")
    parser.add_argument('--no_print_input', dest='print_input', action='store_false', help="Don't print the input alongside the output - gets kind of noisy")
    args = parser.parse_args()

    if len(args.ssurgeon) == 0:
        args.ssurgeon = ["relabelNamedEdge -edge bad -reln advcl"]

    if args.input_file:
        doc = CoNLL.conll2doc(input_file=args.input_file)
    else:
        doc = CoNLL.conll2doc(input_str=SAMPLE_DOC)

    if args.print_input:
        print("{:C}".format(doc))
    ssurgeon_response = process_doc_one_operation(doc, args.semgrex, args.ssurgeon)
    updated_doc = convert_response_to_doc(doc, ssurgeon_response)
    print("{:C}".format(updated_doc))

if __name__ == '__main__':
    main()
