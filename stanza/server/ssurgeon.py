"""Invokes the Java ssurgeon on a document

"ssurgeon" sends text to Java CoreNLP for processing with a ssurgeon
(Semantic graph SURGEON) query

The main program in this file gives a very short intro to how to use it.
"""


import argparse
import copy
import re

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

def parse_ssurgeon_edits(ssurgeon_text):
    ssurgeon_text = ssurgeon_text.strip()
    ssurgeon_blocks = re.split("\n\n+", ssurgeon_text)
    ssurgeon_edits = []
    for idx, block in enumerate(ssurgeon_blocks):
        lines = block.split("\n")
        comments = [line[1:].strip() for line in lines if line.startswith("#")]
        notes = " ".join(comments)
        lines = [x for x in lines if x.strip() and not x.startswith("#")]
        semgrex = lines[0]
        ssurgeon = lines[1:]
        ssurgeon_edits.append(SsurgeonEdit(semgrex, ssurgeon, "%d" % (idx + 1), notes))
    return ssurgeon_edits

def read_ssurgeon_edits(edit_file):
    with open(edit_file, encoding="utf-8") as fin:
        return parse_ssurgeon_edits(fin.read())

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
        # EditNode is currently bugged... :/
        # TODO: change this after next CoreNLP release (after 4.5.3)
        #if not ssurgeon_result.changed:
        #    continue

        ssurgeon_graph = ssurgeon_result.graph
        # TODO: make a script that converts the Pronouns dataset to MWT!
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
                NER: graph_word.ner if graph_word.ner else None,
                MISC: None,
                START_CHAR: None,   # TODO: fix this?  one problem is the text positions
                END_CHAR: None,     #   might change across all of the sentences
                # presumably python will complain if this conflicts
                # with one of the constants above
                "is_mwt": graph_word.isMWT,
                "is_first_mwt": graph_word.isFirstMWT,
                "mwt_text": graph_word.mwtText,
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

        # for any MWT, produce a token_entry which represents the word range
        mwt_tokens = []
        for word_start_idx, word in enumerate(tokens):
            if not word["is_first_mwt"]:
                mwt_tokens.append(word)
                continue
            word_end_idx = word_start_idx + 1
            while word_end_idx < len(tokens) and tokens[word_end_idx]["is_mwt"] and not tokens[word_end_idx]["is_first_mwt"]:
                word_end_idx += 1
            mwt_token_entry = {
                # the tokens don't fencepost the way lists do
                ID: (tokens[word_start_idx][ID], tokens[word_end_idx-1][ID]),
                TEXT: word["mwt_text"],
                NER: word[NER],
                # use the SpaceAfter=No (or not) from the last word in the token
                MISC: tokens[word_end_idx-1][MISC],
            }
            mwt_tokens.append(mwt_token_entry)
            mwt_tokens.append(word)

        old_comments = list(sentence.comments)
        sentence = Sentence(mwt_tokens, doc)

        # TODO: look at word.parent to see if it is part of an MWT
        # once that's done, the beginning words of an MWT do not need SpaceAfter=No any more (it is implied)
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
    parser.add_argument('--edit_file', type=str, default=None, help="File to get semgrex and ssurgeon rules from")
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
    if args.edit_file:
        ssurgeon_edits = read_ssurgeon_edits(args.edit_file)
        ssurgeon_request = build_request(doc, ssurgeon_edits)
        ssurgeon_response = send_ssurgeon_request(ssurgeon_request)
    else:
        ssurgeon_response = process_doc_one_operation(doc, args.semgrex, args.ssurgeon)
    updated_doc = convert_response_to_doc(doc, ssurgeon_response)
    print("{:C}".format(updated_doc))

if __name__ == '__main__':
    main()
