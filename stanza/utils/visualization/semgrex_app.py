import streamlit as st
import streamlit.components.v1 as components
from semgrex_visualizer import visualize_search_str
from semgrex_visualizer import edit_html_overflow
from stanza.utils.conll import CoNLL
import ssurgeon_visualizer as ssv
from stanza.server.ssurgeon import *
from io import StringIO
import os
import stanza
import typing
from typing import List, Tuple, Any
import argparse


def get_text_and_query() -> Tuple[str, str]:
    """
    Gets user input for the Semgrex text and queries to process.

    @return: A tuple containing the user's input text and their input queries
    """
    input_txt = st.text_area(
        "Text to analyze",
        """Banning opal removed artifact decks from the meta.""",
        placeholder="Banning opal removed artifact decks from the meta.",
    )
    input_queries = st.text_area(
        "Semgrex search queries (separate each query with a comma)",
        "{pos:NN}=object <obl {}=action, {cpos:NOUN}=thing <obj {cpos:VERB}=action",
        placeholder="""{pos:NN}=object <obl {}=action, {cpos:NOUN}=thing <obj {cpos:VERB}=action""",
    )
    return input_txt, input_queries


def get_file_input() -> List[str]:
    """
    Allows user to submit files for analysis.

    @return: List of strings containing the file contents of each submitted file. The i-th element of res is the
    string representing the i-th file uploaded.
    """
    st.markdown("""**Alternatively, upload file(s) to analyze.**""")
    uploaded_files = st.file_uploader(
        "button_label", accept_multiple_files=True, label_visibility="collapsed"
    )
    res = []
    for file in uploaded_files:
        stringio = StringIO(file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        res.append(string_data)
    return res


def get_window_input() -> Tuple[bool, int, int]:
    """
    Allows user to specify a specific window of Semgrex hits to visualize. Works similar to Python splicing.

    @return: A tuple containing a bool representing whether or not the user wants to visualize a splice of
    the visualizations, and two ints representing the start and end indices of the splice.
    """
    show_window = st.checkbox(
        "Visualize a specific window of Semgrex search hits?",
        help="""If you want to visualize all search results, leave this unmarked.""",
    )
    start_window, end_window = None, None
    if show_window:
        start_window = st.number_input(
            "Which search hit should visualizations start from?",
            help="""If you want to visualize the first 10 search results, set this to 0.""",
            min_value=0,
        )
        end_window = st.number_input(
            "Which search hit should visualizations stop on?",
            help="""If you want to visualize the first 10 search results, set this to 11.
                                     The 11th result will NOT be displayed.""",
            value=11,
            min_value=start_window + 1,
        )
    return show_window, start_window, end_window


def get_pos_input() -> bool:
    use_xpos = st.checkbox("Would you like to visualize xpos tags?",
                           help="The default visualization options use upos tags for part-of-speech labeling. If xpos tags aren't available for the sentence, displays upos.")
    return use_xpos

def get_input() -> Tuple[str, str, List[str], Tuple[bool, int, int, bool]]:
    input_txt, input_queries = get_text_and_query()
    client_files = get_file_input()  # this is already converted to string format
    window_input = get_window_input()
    visualize_xpos = get_pos_input()
    return input_txt, input_queries, client_files, window_input, visualize_xpos


def run_semgrex_process(
    input_txt: str,
    input_queries: str,
    client_files: List[str],
    show_window: bool,
    clicked: bool,
    pipe: Any,
    start_window: int,
    end_window: int,
    visualize_xpos: bool,
    show_success: bool = True
) -> None:
    """
    Run Semgrex search on the input text/files with input query and serve the HTML on the app.

    @param input_txt: Text to analyze and draw sentences from.
    @param input_queries: Semgrex queries to parse the input with.
    @param client_files: Alternative to input text, we can parse the content of files for scaled analysis.
    @param show_window: Whether or not the user wants a splice of the visualizations
    @param clicked: Whether or not the button has been clicked to run Semgrex search
    @param pipe: NLP pipeline to process input with
    @param start_window: If displaying a splice of visualizations, this is the start idx
    @param end_window: If displaying a splice of visualizations, this is the end idx
    @param visualize_xpos: Set to true if using xpos tags for part of speech labels, otherwise use upos tags

    """

    if clicked:
        if not input_txt and not client_files:
            st.error("Please provide a text input or upload files for analysis.")
        elif input_txt and client_files:
            st.error(
                "Please only choose to visualize your input text or your uploaded files, not both."
            )
        elif not input_queries:
            st.error("Please provide a set of Semgrex queries.")
        else:  # no input errors
            try:
                with st.spinner("Processing..."):
                    queries = [
                        query.strip() for query in input_queries.split(",")
                    ]  # separate queries into individual parts
                    if client_files:
                        html_strings, begin_viz_idx, end_viz_idx = [], 0, float("inf")
                        if show_window:
                            begin_viz_idx, end_viz_idx = (
                                start_window - 1,
                                end_window - 1,
                            )
                        for client_file in client_files:
                            client_file_html_strings = visualize_search_str(
                                client_file,
                                queries,
                                "en",
                                start_match=begin_viz_idx,
                                end_match=end_viz_idx,
                                pipe=pipe,
                                visualize_xpos=visualize_xpos
                            )
                            html_strings += client_file_html_strings
                    else:  # just input text, no files
                        if show_window:
                            html_strings = visualize_search_str(
                                input_txt,
                                queries,
                                "en",
                                start_match=start_window - 1,
                                end_match=end_window - 1,
                                pipe=pipe,
                                visualize_xpos=visualize_xpos
                            )
                        else:
                            html_strings = visualize_search_str(
                                input_txt,
                                queries,
                                "en",
                                end_match=float("inf"),
                                pipe=pipe,
                                visualize_xpos=visualize_xpos
                            )


                    if len(html_strings) == 0:
                        st.write("No Semgrex match hits!")

                    for s in html_strings:
                        s_no_overflow = edit_html_overflow(s)
                        components.html(
                            s_no_overflow, height=200, width=1000, scrolling=True
                        )
                    if show_success:
                        if len(html_strings) == 1:
                            st.success(
                                f"Completed! Visualized {len(html_strings)} Semgrex search hit."
                            )
                        else:
                            st.success(
                                f"Completed! Visualized {len(html_strings)} Semgrex search hits."
                            )
            except OSError:
                st.error(
                    "Your text input or your provided Semgrex queries are incorrect. Please try again."
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--CLASSPATH",
        type=str,
        default=os.environ.get("CLASSPATH"),
        help="""Path to your CoreNLP directory.""",
    )  # for example, set $CLASSPATH to "C:\\stanford-corenlp-4.5.2\\stanford-corenlp-4.5.2\\*"
    args = parser.parse_args()
    print("CLASSPATH:" , args.CLASSPATH)
    CLASSPATH = args.CLASSPATH

    # os.environ["CLASSPATH"] = CLASSPATH
    os.environ["CLASSPATH"] = "C:\\Users\\Alex\\Desktop\\stanford-corenlp-4.5.3\\*"
    if "pipeline" not in st.session_state:  # run pipeline once per user session
        en_nlp_stanza = stanza.Pipeline(
            "en", processors="tokenize, pos, lemma, depparse"
        )
        st.session_state["pipeline"] = en_nlp_stanza

    st.title("Displaying Semgrex Queries")

    html_string = (
        "<h3>Enter a text below, along with your Semgrex query of choice.</h3>"
    )
    st.markdown(html_string, unsafe_allow_html=True)
    input_txt, input_queries, client_files, window_input, visualize_xpos = get_input()

    show_window, start_window, end_window = window_input

    clicked = st.button(
        "Load Semgrex search visualization",
        help="""Semgrex search visualizations only display 
    sentences with a query match. Non-matching sentences are not shown.""",
    )  # use the on_click param

    run_semgrex_process(
        input_txt=input_txt,
        input_queries=input_queries,
        client_files=client_files,
        show_window=show_window,
        clicked=clicked,
        pipe=st.session_state["pipeline"],
        start_window=start_window,
        end_window=end_window,
        visualize_xpos=visualize_xpos
    )

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
    st.title("Displaying Ssurgeon Results")

    input_txt = st.text_area(
        "Text to analyze",
        SAMPLE_DOC,
        placeholder=SAMPLE_DOC,
    )
    semgrex_input_queries = st.text_area(
        "Semgrex search queries (separate each query with a comma)",
        "{}=source >nsubj {} >csubj=bad {}",
        placeholder="""{}=source >nsubj {} >csubj=bad {}""",
    )
    ssurgeon_input_queries = st.text_area(
        "Ssurgeon commands",
        "relabelNamedEdge -edge bad -reln advcl",
        placeholder="relabelNamedEdge -edge bad -reln advcl"
    )

    st.markdown("""**Alternatively, upload file(s) to edit.**""")
    uploaded_files = st.file_uploader(
        "", accept_multiple_files=True, label_visibility="collapsed"
    )
    res = []
    for file in uploaded_files:
        stringio = StringIO(file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        res.append(string_data)

    clicked = st.button(
        "Load Semgrex search visualization",
        help="""Semgrex search visualizations only display 
        sentences with a query match. Non-matching sentences are not shown.""",
    )
    clicked_for_file_edit = st.button(
        "Edit File"
    )

    if clicked:
        try:
            with st.spinner("Processing..."):
                semgrex_queries = semgrex_input_queries # separate queries into individual parts
                ssurgeon_queries = [ssurgeon_input_queries]
                html_strings = ssv.visualize_edited_deprel_adjusted_str_input(input_txt, semgrex_queries, ssurgeon_queries)
                doc = CoNLL.conll2doc(input_str=input_txt)
                string_txt = " ".join([word.text for sentence in doc.sentences for word in sentence.words])

                html_string = (
                    "<h3>Previous deprel visualization:</h3>"
                )
                st.markdown(html_string, unsafe_allow_html=True)
                components.html(
                    run_semgrex_process(input_txt=string_txt, input_queries=semgrex_queries, clicked=clicked,
                                        show_window=False, client_files=[], pipe=st.session_state["pipeline"],
                                        start_window=1, end_window=11, visualize_xpos=visualize_xpos, show_success=False)
                )

                if len(html_strings) == 0:
                    st.write("No Semgrex match hits!")

                for s in html_strings:
                    html_string = (
                        "<h3>Edited deprel visualization:</h3>"
                    )
                    st.markdown(html_string, unsafe_allow_html=True)
                    s_no_overflow = edit_html_overflow(s)
                    components.html(
                        s_no_overflow, height=200, width=1000, scrolling=True
                    )
        except OSError:
            st.error(
                "Your text input or your provided Semgrex/Ssurgeon queries are incorrect. Please try again."
            )
    if clicked_for_file_edit:
        # files are in res
        if len(res) == 0:
            st.error("You must provide files for analysis.")
        with st.spinner("Editing..."):
            single_file = res[0]
            doc = CoNLL.conll2doc(input_str=single_file)
            ssurgeon_response = process_doc_one_operation(doc, semgrex_input_queries, [ssurgeon_input_queries])
            updated_doc = convert_response_to_doc(doc, ssurgeon_response)
            output = CoNLL.doc2conll(updated_doc)[0]
            output_str = "\n".join(output)
            st.download_button("Download your edited file", data=output_str, file_name="SSurgeon.conll")

if __name__ == "__main__":
    main()
