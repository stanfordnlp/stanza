"""
Visualization tooling for Ssurgeon
"""
import os
import sys
import stanza.utils.visualization.semgrex_visualizer as sv
import stanza.server.ssurgeon
from stanza.server.ssurgeon import process_doc_one_operation, convert_response_to_doc
from stanza.utils.conll import CoNLL
from stanza.utils.visualization.constants import *
import logging


def generate_edited_deprel_unadjusted(edited_doc, lang_code, visualize_xpos):
    """
    Submit edited doc from ssurgeon to generate HTML for sentences output
    :param edited_doc:
    :param lang_code:
    :param visualize_xpos:
    :return:
    """
    return sv.get_sentences_html(doc=edited_doc, language=lang_code, visualize_xpos=visualize_xpos)


def visualize_ssurgeon_deprel_adjusted_str_input(input_str, semgrex_query, ssurgeon_query, lang_code="en", visualize_xpos=False, render=False):
    """
    Visualizes the edited side of the ssurgeon edit
    :param unedited_doc:
    :param semgrex_query:
    :param ssurgeon_query:
    :return:
    """
    doc = CoNLL.conll2doc(input_str=input_str)
    ssurgeon_response = process_doc_one_operation(doc, semgrex_query, ssurgeon_query)
    updated_doc = convert_response_to_doc(doc, ssurgeon_response)
    html_strings = generate_edited_deprel_unadjusted(updated_doc, lang_code, visualize_xpos=visualize_xpos)
    edited_html_strings = []
    for i in range(len(html_strings)):
        edited_html = sv.adjust_dep_arrows(html_strings[i])
        edited_html_strings.append(edited_html)

    if render:
        sv.render_html_strings(edited_html_strings)

    return edited_html_strings


def main():
    # Load classpath if not already existing
    if not os.environ.get('CLASSPATH'):
        logging.info("Load the path to wherever CoreNLP is installed on your machine to $CLASSPATH.")

    # The default semgrex detects sentences in the UD_English-Pronouns dataset which have both nsubj and csubj on the same word.
    # The default ssurgeon transforms the unwanted csubj to advcl
    # See https://github.com/UniversalDependencies/docs/issues/923
    ssurgeon = ["relabelNamedEdge -edge bad -reln advcl"]  # example
    semgrex = "{}=source >nsubj {} >csubj=bad {}"  # example
    SSURGEON_JAVA = "edu.stanford.nlp.semgraph.semgrex.ssurgeon.ProcessSsurgeonRequest"
    doc = CoNLL.conll2doc(input_str=SAMPLE_SSURGEON_DOC)

    print("{:C}".format(doc))
    ssurgeon_response = process_doc_one_operation(doc, semgrex, ssurgeon)
    updated_doc = convert_response_to_doc(doc, ssurgeon_response)
    print("{:C}".format(updated_doc))
    print(generate_edited_deprel_unadjusted(updated_doc, lang_code='en', visualize_xpos=False))
    visualize_ssurgeon_deprel_adjusted_str_input(SAMPLE_SSURGEON_DOC, semgrex, ssurgeon)


if __name__ == '__main__':
    main()
