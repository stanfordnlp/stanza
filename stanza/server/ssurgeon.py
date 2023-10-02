"""Invokes the Java ssurgeon on a document

"ssurgeon" sends text to Java CoreNLP for processing with a ssurgeon
(Semantic graph SURGEON) query

The main program in this file gives a very short intro to how to use it.
"""


import argparse
import copy
import os
import re
import sys

from stanza.protobuf import SsurgeonRequest, SsurgeonResponse
from stanza.server import java_protobuf_requests
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
        lines = [x.strip() for x in lines if x.strip() and not x.startswith("#")]
        if len(lines) == 0:
            # was a block of entirely comments
            continue
        semgrex = lines[0]
        ssurgeon = lines[1:]
        ssurgeon_edits.append(SsurgeonEdit(semgrex, ssurgeon, "%d" % (idx + 1), notes))
    return ssurgeon_edits

def read_ssurgeon_edits(edit_file):
    with open(edit_file, encoding="utf-8") as fin:
        return parse_ssurgeon_edits(fin.read())

def send_ssurgeon_request(request):
    return java_protobuf_requests.send_request(request, SsurgeonResponse, SSURGEON_JAVA)

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

    try:
        for sent_idx, sentence in enumerate(doc.sentences):
            graph = request.graph.add()
            word_idx = 0
            for token in sentence.tokens:
                for word in token.words:
                    java_protobuf_requests.add_token(graph.token, word, token)
                    java_protobuf_requests.add_word_to_graph(graph, word, sent_idx, word_idx)

                    word_idx = word_idx + 1
    except Exception as e:
        raise RuntimeError("Failed to process sentence {}:\n{:C}".format(sent_idx, sentence)) from e

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
    try:
        for sent_idx, (sentence, ssurgeon_result) in enumerate(zip(doc.sentences, semgrex_response.result)):
            # EditNode is currently bugged... :/
            # TODO: change this after next CoreNLP release (after 4.5.3)
            #if not ssurgeon_result.changed:
            #    continue

            ssurgeon_graph = ssurgeon_result.graph
            tokens = []
            for graph_node, graph_word in zip(ssurgeon_graph.node, ssurgeon_graph.token):
                if graph_node.copyAnnotation:
                    continue
                word_entry = {
                    ID: graph_node.index,
                    TEXT: graph_word.word if graph_word.word else None,
                    LEMMA: graph_word.lemma if graph_word.lemma else None,
                    UPOS: graph_word.coarseTag if graph_word.coarseTag else None,
                    XPOS: graph_word.pos if graph_word.pos else None,
                    FEATS: java_protobuf_requests.features_to_string(graph_word.conllUFeatures),
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
                    "mwt_misc": graph_word.mwtMisc,
                }
                # TODO: do "before" as well
                word_entry[MISC] = java_protobuf_requests.space_after_to_misc(graph_word.after)
                if graph_word.conllUMisc:
                    word_entry[MISC] = java_protobuf_requests.substitute_space_misc(graph_word.conllUMisc, word_entry[MISC])
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
                    if word["is_mwt"]:
                        word[MISC] = java_protobuf_requests.remove_space_misc(word[MISC])
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
                    MISC: None,
                }
                mwt_token_entry[MISC] = java_protobuf_requests.misc_space_pieces(tokens[word_end_idx-1][MISC])
                if tokens[word_end_idx-1]["mwt_misc"]:
                    mwt_token_entry[MISC] = java_protobuf_requests.substitute_space_misc(tokens[word_end_idx-1]["mwt_misc"], mwt_token_entry[MISC])
                word[MISC] = java_protobuf_requests.remove_space_misc(word[MISC])
                mwt_tokens.append(mwt_token_entry)
                mwt_tokens.append(word)

            old_comments = list(sentence.comments)
            sentence = Sentence(mwt_tokens, doc)

            token_text = [token.text if (token_idx == len(sentence.tokens) - 1 or
                                         (token.misc and "SpaceAfter=No" in token.misc.split("|")) or
                                         (token.words[-1].misc and "SpaceAfter=No" in token.words[-1].misc.split("|")))
                         else token.text + " "
                         for token_idx, token in enumerate(sentence.tokens)]
            sentence_text = "".join(token_text)

            for comment in old_comments:
                if comment.startswith("# text ") or comment.startswith("#text ") or comment.startswith("# text=") or comment.startswith("#text="):
                    sentence.add_comment("# text = " + sentence_text)
                else:
                    sentence.add_comment(comment)

            doc.sentences[sent_idx] = sentence

            sentence.rebuild_dependencies()
    except Exception as e:
        raise RuntimeError("Ssurgeon could not process sentence {}\nSsurgeon result:\n{}\nOriginal sentence:\n{:C}".format(sent_idx, ssurgeon_result, sentence)) from e
    return doc

class Ssurgeon(java_protobuf_requests.JavaProtobufContext):
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
    # for Windows, so that we aren't randomly printing garbage (or just failing to print)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # TODO: deprecate 3.6 support after the next release
        pass

    # The default semgrex detects sentences in the UD_English-Pronouns dataset which have both nsubj and csubj on the same word.
    # The default ssurgeon transforms the unwanted csubj to advcl
    # See https://github.com/UniversalDependencies/docs/issues/923
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=None, help="Input file to process (otherwise will process a sample text)")
    parser.add_argument('--output_file', type=str, default=None, help="Output file (otherwise will write to stdout)")
    parser.add_argument('--input_dir', type=str, default=None, help="Input dir to process instead of a single file.  Allows for reusing the Java program")
    parser.add_argument('--input_filter', type=str, default=".*[.]conllu", help="Only process files from the input_dir that match this filter - regex, not shell filter.  Default: %(default)s")
    parser.add_argument('--no_input_filter', action='store_const', const=None, help="Remove the default input filename filter")
    parser.add_argument('--output_dir', type=str, default=None, help="Output dir for writing files, necessary if using --input_dir")
    parser.add_argument('--edit_file', type=str, default=None, help="File to get semgrex and ssurgeon rules from")
    parser.add_argument('--semgrex', type=str, default="{}=source >nsubj {} >csubj=bad {}", help="Semgrex to apply to the text.  A default detects words which have both an nsubj and a csubj.  Default: %(default)s")
    parser.add_argument('ssurgeon', type=str, default=["relabelNamedEdge -edge bad -reln advcl"], nargs="*", help="Ssurgeon edits to apply based on the Semgrex.  Can have multiple edits in a row.  A default exists to transform csubj into advcl.  Default: %(default)s")
    parser.add_argument('--print_input', dest='print_input', action='store_true', default=False, help="Print the input alongside the output - gets kind of noisy.  Default: %(default)s")
    parser.add_argument('--no_print_input', dest='print_input', action='store_false', help="Don't print the input alongside the output - gets kind of noisy")
    args = parser.parse_args()

    if args.edit_file:
        ssurgeon_edits = read_ssurgeon_edits(args.edit_file)
    else:
        ssurgeon_edits = [SsurgeonEdit(args.semgrex, args.ssurgeon)]

    if args.input_file:
        docs = [CoNLL.conll2doc(input_file=args.input_file)]
        outputs = [args.output_file]
        input_output = zip(docs, outputs)
    elif args.input_dir:
        if not args.output_dir:
            raise ValueError("Cannot process multiple files without knowing where to send them - please set --output_dir in order to use --input_dir")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        def read_docs():
            for doc_filename in os.listdir(args.input_dir):
                if args.input_filter:
                    if not re.match(args.input_filter, doc_filename):
                        continue
                doc_path = os.path.join(args.input_dir, doc_filename)
                output_path = os.path.join(args.output_dir, doc_filename)
                print("Processing %s to %s" % (doc_path, output_path))
                yield CoNLL.conll2doc(input_file=doc_path), output_path
        input_output = read_docs()
    else:
        docs = [CoNLL.conll2doc(input_str=SAMPLE_DOC)]
        outputs = [None]
        input_output = zip(docs, outputs)

    for doc, output in input_output:
        if args.print_input:
            print("{:C}".format(doc))
        ssurgeon_request = build_request(doc, ssurgeon_edits)
        ssurgeon_response = send_ssurgeon_request(ssurgeon_request)
        updated_doc = convert_response_to_doc(doc, ssurgeon_response)
        if output is not None:
            with open(output, "w", encoding="utf-8") as fout:
                fout.write("{:C}\n\n".format(updated_doc))
        else:
            print("{:C}\n".format(updated_doc))

if __name__ == '__main__':
    main()
