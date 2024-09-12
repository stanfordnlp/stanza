import os
import argparse
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

from stanza.pipeline.core import Pipeline
from stanza.server.semgrex import Semgrex
from stanza.models.common.constant import is_right_to_left
import spacy
from spacy import displacy
from spacy.tokens import Doc
from IPython.display import display, HTML
import typing
from typing import List, Tuple, Any

from stanza.utils.visualization.utils import find_nth, round_base


def get_sentences_html(doc: Any, language: str, visualize_xpos: bool = False) -> List[str]:
    """
    Returns a list of HTML strings representing the dependency visualizations of a given stanza document.
    One HTML string is generated per sentence of the document object. Converts the stanza document object
    to a spaCy doc object and generates HTML with displaCy.

    @param doc: a stanza document object which can be generated with an NLP pipeline.
    @param language: the two letter language code for the document e.g. "en" for English.
    @param visualize_xpos: A toggled option to use xpos tags for part-of-speech labels instead of upos.

    @return: a list of HTML strings which visualize the dependencies of the doc object.
    """
    USE_FINE_GRAINED = False if not visualize_xpos else True
    html_strings, sentences_to_visualize = [], []
    nlp = spacy.blank(
        "en"
    )  # blank model - we don't use any of the model features, just the visualization
    for sentence in doc.sentences:
        words, lemmas, heads, deps, tags = [], [], [], [], []
        if is_right_to_left(
            language
        ):  # order of words displayed is reversed, dependency arcs remain intact
            sentence_len = len(sentence.words)
            for word in reversed(sentence.words):
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                if visualize_xpos and word.xpos:
                    tags.append(word.xpos)
                else:
                    tags.append(word.upos)
                if word.head == 0:  # spaCy head indexes are one-off from Stanza's
                    heads.append(sentence_len - word.id)
                else:
                    heads.append(sentence_len - word.head)
        else:  # left to right rendering
            for word in sentence.words:
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                if visualize_xpos and word.xpos:
                    tags.append(word.xpos)
                else:
                    tags.append(word.upos)
                if word.head == 0:
                    heads.append(word.id - 1)
                else:
                    heads.append(word.head - 1)
        if USE_FINE_GRAINED:
            stanza_to_spacy_doc = Doc(
                nlp.vocab, words=words, lemmas=lemmas, heads=heads, deps=deps, tags=tags
            )
        else:
            stanza_to_spacy_doc = Doc(
                nlp.vocab, words=words, lemmas=lemmas, heads=heads, deps=deps, pos=tags
            )
        sentences_to_visualize.append(stanza_to_spacy_doc)

    for line in sentences_to_visualize:  # render all sentences through displaCy
        html_strings.append(
            displacy.render(
                line,
                style="dep",
                options={
                    "compact": True,
                    "word_spacing": 30,
                    "distance": 100,
                    "arrow_spacing": 20,
                    "fine_grained": USE_FINE_GRAINED
                },
                jupyter=False,
            )
        )
    return html_strings


def semgrexify_html(orig_html: str, semgrex_sentence) -> str:
    """
    Modifies the HTML of a sentence's dependency visualization, highlighting words involved in the
    semgrex_sentence search queries and adding the label of the word inside of the match.


    @param orig_html: unedited HTML of a sentence's dependency visualization.
    @param semgrex_sentence: a Semgrex result object containing the matches to a provided query.
    @return: edited HTML containing the visual changes described above.
    """
    tracker = {}  # keep track of which words have multiple labels
    DEFAULT_TSPAN_COUNT = (
        2  # the original displacy html assigns two <tspan> objects per <text> object
    )
    CLOSING_TSPAN_LEN = 8  # </tspan> is 8 chars long
    colors = [
        "#4477AA",
        "#66CCEE",
        "#228833",
        "#CCBB44",
        "#EE6677",
        "#AA3377",
        "#BBBBBB",
    ]  # colorblind-friendly scheme
    css_bolded_class = "<style> .bolded{font-weight: bold;} </style>\n"
    opening_svg_end_idx = orig_html.find("\n")
    # insert the new style class
    orig_html = (
        orig_html[: opening_svg_end_idx + 1]
        + css_bolded_class
        + orig_html[opening_svg_end_idx + 1 :]
    )

    # Color and bold words involved in each Semgrex match
    for query in semgrex_sentence.result:
        for i, match in enumerate(query.match):
            color = colors[i]
            paired_dy = 2
            for node in match.node:
                name, match_index = node.name, node.matchIndex
                # edit existing <tspan> to change color and bold the text
                start = find_nth(
                    orig_html, "<text", match_index
                )  # finds start of svg <text> of interest
                if (
                    match_index not in tracker
                ):  # if we've already bolded and colored, keep the first color
                    tspan_start = orig_html.find(
                        "<tspan", start
                    )  # finds start of the first svg <tspan> inside of the <text>
                    tspan_end = orig_html.find(
                        "</tspan>", start
                    )  # finds start of the end of the above <tspan>
                    tspan_substr = (
                        orig_html[tspan_start : tspan_end + CLOSING_TSPAN_LEN + 1]
                        + "\n"
                    )
                    # color and bold words in the search hit
                    edited_tspan = tspan_substr.replace(
                        'class="displacy-word"', 'class="bolded"'
                    ).replace('fill="currentColor"', f'fill="{color}"')
                    # insert edited <tspan> object into html string

                    # TODO: DEBUG. This code has a bug in it that causes the svg to not end on an input like
                    # "The Wimbledon grass-court tennis tournament banned players, resulting in players hating others."
                    # to malfunction and add another <svg> copy to the tail-end of the first svg rendering.
                    # This bug has been patched in the end of this function, but need to find out what is going on.
                    orig_html = (
                        orig_html[:tspan_start]
                        + edited_tspan
                        + orig_html[tspan_end + CLOSING_TSPAN_LEN + 2 :]
                    )

                    tracker[match_index] = DEFAULT_TSPAN_COUNT

                # next, we have to insert the new <tspan> object for the label
                # Copy old <tspan> to copy formatting when creating new <tspan> later
                prev_tspan_start = (
                    find_nth(orig_html[start:], "<tspan", tracker[match_index] - 1)
                    + start
                )  # find the previous <tspan> start index
                prev_tspan_end = (
                    find_nth(orig_html[start:], "</tspan>", tracker[match_index] - 1)
                    + start
                )  # find the prev </tspan> start index
                prev_tspan = orig_html[
                    prev_tspan_start : prev_tspan_end + CLOSING_TSPAN_LEN + 1
                ]

                # Find spot to insert new tspan
                closing_tspan_start = (
                    find_nth(orig_html[start:], "</tspan>", tracker[match_index])
                    + start
                )
                up_to_new_tspan = orig_html[
                    : closing_tspan_start + CLOSING_TSPAN_LEN + 1
                ]
                rest = orig_html[closing_tspan_start + CLOSING_TSPAN_LEN + 1 :]

                # Calculate proper x value in svg
                x_value_start = prev_tspan.find('x="')
                x_value_end = (
                    prev_tspan[x_value_start + 3 :].find('"') + 3
                )  # 3 is the length of the 'x="' substring
                x_value = prev_tspan[x_value_start + 3 : x_value_end + x_value_start]

                # Calculate proper y value in svg
                DEFAULT_DY_VAL, dy = 2, 2
                if (
                    paired_dy != DEFAULT_DY_VAL and node == match.node[1]
                ):  # we're on the second node and need to adjust height to match the paired node
                    dy = paired_dy
                if node == match.node[0] and len(match.node) > 1:
                    paired_node_level = 2
                    if (
                        match.node[1].matchIndex in tracker
                    ):  # check if we need to adjust heights of labels
                        paired_node_level = tracker[match.node[1].matchIndex]
                        dif = tracker[match_index] - paired_node_level
                        if dif > 0:  # current node has more labels
                            paired_dy = DEFAULT_DY_VAL * dif + 1
                            dy = DEFAULT_DY_VAL
                        else:  # paired node has more labels, adjust this label down
                            dy = DEFAULT_DY_VAL * (abs(dif) + 1)
                            paired_dy = DEFAULT_DY_VAL

                # Insert new <tspan> object
                new_tspan = f'  <tspan class="displacy-word" dy="{dy}em" fill="{color}" x={x_value}>{name[: 3].title()}.</tspan>\n'  # abbreviate label names to 3 chars
                orig_html = up_to_new_tspan + new_tspan + rest
                tracker[match_index] += 1

        # process out extra term if present -- TODO: Figure out why the semgrexify_html function lines 164-168 cause a duplication bug
        end = find_nth(haystack=orig_html, needle="</svg", n=1)
        LENGTH_OF_END_SVG = 7  # </svg> has length 6 so add 1 to the end too
        if len(orig_html) > end + LENGTH_OF_END_SVG:
            orig_html = orig_html[: end + LENGTH_OF_END_SVG]

    return orig_html


def render_html_strings(edited_html_strings: List[str]) -> None:
    """
    Renders the HTML of each HTML string.
    """
    for html_string in edited_html_strings:
        display(HTML(html_string))


def visualize_search_doc(
    doc: Any,
    semgrex_queries: List[str],
    lang_code: str,
    start_match: int = 0,
    end_match: int = 11,
    render: bool = True,
    visualize_xpos: bool = False
) -> List[str]:
    """
    Visualizes the result of running Semgrex search on a document. The i-th element of
    the returned list is the HTML representation of the i-th sentence's dependency
    relationships. Only shows sentences that have a match on the Semgrex search.

    @param doc: A Stanza document object that contains dependency relationships .
    @param semgrex_queries: A list of Semgrex queries to search for in the document.
    @param lang_code: A two letter language abbreviation for the language that the Stanza document is written in.
    @param start_match: Beginning of the splice for which to display elements with.
    @param end_match: End of the splice for which to display elements with.
    @param render: A toggled option to render the HTML strings within the returned list
    @param visualize_xpos: A toggled option to use xpos tags in part-of-speech labels, defaulting to upos tags.

    @return: A list of HTML strings representing the dependency relations of the doc object.
    """

    matches_count = 0  # Limits number of visualizations
    with Semgrex(classpath="$CLASSPATH") as sem:
        edited_html_strings = []
        semgrex_results = sem.process(doc, *semgrex_queries)
        # one html string for each sentence
        unedited_html_strings = get_sentences_html(doc, lang_code, visualize_xpos=visualize_xpos)

        for i in range(len(unedited_html_strings)):

            if matches_count >= end_match:  # we've collected enough matches
                break

            # check if sentence has matches, if not then do not visualize
            has_none = True
            for query in semgrex_results.result[i].result:
                for match in query.match:
                    if match:
                        has_none = False

            # Process HTML if queries have matches
            if not has_none:
                if start_match <= matches_count < end_match:
                    edited_string = semgrexify_html(
                        unedited_html_strings[i], semgrex_results.result[i]
                    )

                    edited_string = adjust_dep_arrows(edited_string)
                    edited_html_strings.append(edited_string)
                matches_count += 1
        if render:
            render_html_strings(edited_html_strings)
    return edited_html_strings


def visualize_search_str(
    text: str,
    semgrex_queries: List[str],
    lang_code: str,
    start_match: int = 0,
    end_match: int = 11,
    pipe=None,
    render: bool = True,
    visualize_xpos: bool = False
):
    """
    Visualizes the result of running Semgrex search on a string. The i-th element of
    the returned list is the HTML representation of the i-th sentence's dependency
    relationships. Only shows sentences that have a match on the Semgrex search.

    @param text: The string for which Semgrex search will be run on.
    @param semgrex_queries: A list of Semgrex queries to search for in the document.
    @param lang_code: A two letter language abbreviation for the language that the Stanza document is written in.
    @param start_match: Beginning of the splice for which to display elements with.
    @param end_match: End of the splice for which to display elements with.
    @param pipe: An NLP pipeline through which the text will be processed.
    @param render: A toggled option to render the HTML strings within the returned list.
    @param visualize_xpos: A toggled option to use xpos tags for part-of-speech labeling, defaulting to upos tags

    @return: A list of HTML strings representing the dependency relations of the doc object.
    """
    if pipe is None:
        nlp = Pipeline(lang_code, processors="tokenize, pos, lemma, depparse")
    else:
        nlp = pipe
    doc = nlp(text)
    return visualize_search_doc(
        doc,
        semgrex_queries,
        lang_code,
        start_match=start_match,
        end_match=end_match,
        render=render,
        visualize_xpos=visualize_xpos
    )


def adjust_dep_arrows(raw_html: str) -> str:
    """
    Default spaCy dependency visualizations have misaligned arrows. Fix arrows by aligning arrow ends and bodies
    to the word that they are directed to.

    @param raw_html: Dependency relation visualization generated HTML from displaCy
    @return: Edited HTML string with fixed arrow placements
    """

    HTML_ARROW_BEGINNING = '<g class="displacy-arrow">'
    HTML_ARROW_ENDING = "</g>"
    HTML_ARROW_ENDING_LEN = 6  # there are 2 newline chars after the arrow ending
    arrows_start_idx = find_nth(
        haystack=raw_html, needle='<g class="displacy-arrow">', n=1
    )
    words_html, arrows_html = (
        raw_html[:arrows_start_idx],
        raw_html[arrows_start_idx:],
    )  # separate html for words and arrows
    final_html = (
        words_html  # continually concatenate to this after processing each arrow
    )
    arrow_number = 1  # which arrow we're currently editing (1-indexed)
    start_idx, end_of_class_idx = (
        find_nth(haystack=arrows_html, needle=HTML_ARROW_BEGINNING, n=arrow_number),
        find_nth(haystack=arrows_html, needle=HTML_ARROW_ENDING, n=arrow_number),
    )
    while start_idx != -1:  # edit every arrow
        arrow_section = arrows_html[
            start_idx : end_of_class_idx + HTML_ARROW_ENDING_LEN
        ]  # slice a single svg arrow object
        if (
            arrow_section[-1] == "<"
        ):  # this is the last arrow in the HTML, don't cut the splice early
            arrow_section = arrows_html[start_idx:]
        edited_arrow_section = edit_dep_arrow(arrow_section)

        final_html = (
            final_html + edited_arrow_section
        )  # continually update html with new arrow html until done

        # Prepare for next iteration
        arrow_number += 1
        start_idx = find_nth(arrows_html, '<g class="displacy-arrow">', arrow_number)
        end_of_class_idx = find_nth(arrows_html, "</g>", arrow_number)
    return final_html


def edit_dep_arrow(arrow_html: str) -> str:
    """
    The formatting of a single displacy arrow in svg is the following:
    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-c628889ffbf343e3848193a08606f10a-0-0" stroke-width="2px" d="M70,352.0 C70,177.0 390.0,177.0 390.0,352.0" fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-c628889ffbf343e3848193a08606f10a-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">csubj</textPath>
        </text>
        <path class="displacy-arrowhead" d="M70,354.0 L62,342.0 78,342.0" fill="currentColor"/>
    </g>

    We edit the 'd = ...' parts of the <path class ...> section to fix the arrow direction and length to round to
    the nearest 50 units, centering on each word's center. This is because the words start at x=50 and have spacing
    of 100, so each word is at an x-value that is a multiple of 50.

    @param arrow_html: Original SVG for a single displaCy arrow.
    @return: Edited SVG for the displaCy arrow, adjusting its placement
    """

    WORD_SPACING = 50  # words start at x=50 and are separated by 100s so their x values are multiples of 50
    M_OFFSET = 4  # length of 'd="M' that we search for to extract the number from d="M70, for instance
    ARROW_PIXEL_SIZE = 4
    first_d_idx, second_d_idx = (
        find_nth(arrow_html, 'd="M', 1),
        find_nth(arrow_html, 'd="M', 2),
    )  # find where d="M starts
    first_d_cutoff, second_d_cutoff = (
        arrow_html.find(",", first_d_idx),
        arrow_html.find(",", second_d_idx),
    )  # isolate the number after 'M' e.g. 'M70'
    # gives svg x values of arrow body starting position and arrowhead position
    arrow_position, arrowhead_position = (
        float(arrow_html[first_d_idx + M_OFFSET : first_d_cutoff]),
        float(arrow_html[second_d_idx + M_OFFSET : second_d_cutoff]),
    )
    # gives starting index of where 'fill="none"' or 'fill="currentColor"' begin, reference points to end the d= section
    first_fill_start_idx, second_fill_start_idx = (
        find_nth(arrow_html, "fill", n=1),
        find_nth(arrow_html, "fill", n=3),
    )

    # isolate the d= ... section to edit
    first_d, second_d = (
        arrow_html[first_d_idx:first_fill_start_idx],
        arrow_html[second_d_idx:second_fill_start_idx],
    )
    first_d_split, second_d_split = first_d.split(","), second_d.split(",")

    if (
        arrow_position == arrowhead_position
    ):  # This arrow is incoming onto the word, center the arrow/head to word center
        corrected_arrow_pos = corrected_arrowhead_pos = round_base(
            arrow_position, base=WORD_SPACING
        )

        # edit first_d  -- arrow body
        second_term = first_d_split[1].split(" ")[0] + " " + str(corrected_arrow_pos)
        first_d = (
            'd="M'
            + str(corrected_arrow_pos)
            + ","
            + second_term
            + ","
            + ",".join(first_d_split[2:])
        )

        # edit second_d  -- arrowhead
        second_term = (
            second_d_split[1].split(" ")[0]
            + " L"
            + str(corrected_arrowhead_pos - ARROW_PIXEL_SIZE)
        )
        third_term = (
            second_d_split[2].split(" ")[0]
            + " "
            + str(corrected_arrowhead_pos + ARROW_PIXEL_SIZE)
        )
        second_d = (
            'd="M'
            + str(corrected_arrowhead_pos)
            + ","
            + second_term
            + ","
            + third_term
            + ","
            + ",".join(second_d_split[3:])
        )
    else:  # This arrow is outgoing to another word, center the arrow/head to that word's center
        corrected_arrowhead_pos = round_base(arrowhead_position, base=WORD_SPACING)

        # edit first_d -- arrow body
        third_term = first_d_split[2].split(" ")[0] + " " + str(corrected_arrowhead_pos)
        fourth_term = (
            first_d_split[3].split(" ")[0] + " " + str(corrected_arrowhead_pos)
        )
        terms = [
            first_d_split[0],
            first_d_split[1],
            third_term,
            fourth_term,
        ] + first_d_split[4:]
        first_d = ",".join(terms)

        # edit second_d -- arrow head
        first_term = f'd="M{corrected_arrowhead_pos}'
        second_term = (
            second_d_split[1].split(" ")[0]
            + " L"
            + str(corrected_arrowhead_pos - ARROW_PIXEL_SIZE)
        )
        third_term = (
            second_d_split[2].split(" ")[0]
            + " "
            + str(corrected_arrowhead_pos + ARROW_PIXEL_SIZE)
        )
        terms = [first_term, second_term, third_term] + second_d_split[3:]
        second_d = ",".join(terms)
    # rebuild and return html from its individual sections
    return (
        arrow_html[:first_d_idx]
        + first_d
        + " "
        + arrow_html[first_fill_start_idx:second_d_idx]
        + second_d
        + " "
        + arrow_html[second_fill_start_idx:]
    )


def edit_html_overflow(html_string: str) -> str:
    """
    Adds to overflow and display settings to the SVG header to visualize overflowing HTML renderings in the
    Semgrex streamlit app. Prevents Semgrex search tags from being cut off at the bottom of visualizations.

    The opening of each HTML string looks similar to this; we add to the end of the SVG header.

    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="fa9446a525de4862b233007f26dbbecb-0" class="displacy" width="850" height="242.0" direction="ltr" style="max-width: none; height: 242.0px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr">
    <style> .bolded{font-weight: bold;} </style>
    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="182.0">
        <tspan class="bolded" fill="#66CCEE" x="50">Banning</tspan>

       <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">VERB</tspan>
      <tspan class="displacy-word" dy="2em" fill="#66CCEE" x=50>Act.</tspan>
    </text>

    @param html_string: HTML of the result of running Semgrex search on a text
    @return: Edited HTML to visualize the dependencies even in the case of overflow.
    """

    BUFFER_LEN = 14  # length of 'direction: ltr"'
    editing_start_idx = find_nth(html_string, "direction: ltr", n=1)
    SVG_HEADER_ADDITION = "overflow: visible; display: block"
    return (
        html_string[:editing_start_idx]
        + "; "
        + SVG_HEADER_ADDITION
        + html_string[editing_start_idx + BUFFER_LEN :]
    )


def main():
    """
    IMPORTANT: For the code in this module to run, you must have corenlp and Java installed on your machine. Additionally,
    set an environment variable CLASSPATH equal to the path of your corenlp directory.

    Example: CLASSPATH=C:\\Users\\Alex\\PycharmProjects\\pythonProject\\stanford-corenlp-4.5.0\\stanford-corenlp-4.5.0\\*
    """
    nlp = Pipeline("en", processors="tokenize,pos,lemma,depparse")
    doc = nlp(
        "Banning opal removed artifact decks from the meta. Banning tennis resulted in players banning people."
    )
    queries = [
        "{pos:NN}=object <obl {}=action",
        "{cpos:NOUN}=thing <obj {cpos:VERB}=action",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc", type=Any, default=doc, help="Stanza document to process."
    )
    parser.add_argument(
        "--queries",
        type=List[str],
        default=queries,
        help="Semgrex queries to search for",
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        default="en",
        help="Two letter abbreviation the document's language e.g. 'en' for English",
    )
    parser.add_argument(
        "--CLASSPATH",
        type=str,
        default="C:\\stanford-corenlp-4.5.2\\stanford-corenlp-4.5.2\\*",
        help="Path to your coreNLP directory",
    )
    args = parser.parse_args()
    os.environ["CLASSPATH"] = args.CLASSPATH
    try:
        res = visualize_search_doc(doc, queries, "en")
        print(res[0])  # first sentence's HTML
    except TypeError:
        raise TypeError(
            """For the code in this module to run, you must have corenlp and Java installed on your machine. 
            Once installed, you can pass in the path to your corenlp directory as a command-line argument named 
            "CLASSPATH". Alternatively, set an environment variable CLASSPATH equal to the path of your corenlp 
            directory."""
        )
    return


if __name__ == "__main__":
    main()
