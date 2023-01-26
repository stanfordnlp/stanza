import stanza
from stanza.server.semgrex import Semgrex
from stanza.models.common.constant import is_right_to_left
import spacy
from spacy import displacy
from spacy.tokens import Doc
from IPython.display import display, HTML


"""
IMPORTANT: For the code in this module to run, you must have corenlp and Java installed on your machine. Additionally,
set an environment variable CLASSPATH equal to the path of your corenlp directory.

Example: CLASSPATH=C:\\Users\\Alex\\PycharmProjects\\pythonProject\\stanford-corenlp-4.5.0\\stanford-corenlp-4.5.0\\*
"""


def get_sentences_html(doc, language):
    """
    Returns a list of the HTML strings of the dependency visualizations of a given stanza doc object.

    The 'language' arg is the two-letter language code for the document to be processed.

    First converts the stanza doc object to a spacy doc object and uses displacy to generate an HTML
    string for each sentence of the doc object.
    """
    html_strings = []

    # blank model - we don't use any of the model features, just the visualization
    nlp = spacy.blank("en")
    sentences_to_visualize = []
    for sentence in doc.sentences:
        words, lemmas, heads, deps, tags = [], [], [], [], []
        if is_right_to_left(language):  # order of words displayed is reversed, dependency arcs remain intact
            sent_len = len(sentence.words)
            for word in reversed(sentence.words):
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                tags.append(word.upos)
                if word.head == 0:  # spaCy head indexes are formatted differently than that of Stanza
                    heads.append(sent_len - word.id)
                else:
                    heads.append(sent_len - word.head)
        else:  # left to right rendering
            for word in sentence.words:
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                tags.append(word.upos)
                if word.head == 0:
                    heads.append(word.id - 1)
                else:
                    heads.append(word.head - 1)
        document_result = Doc(nlp.vocab, words=words, lemmas=lemmas, heads=heads, deps=deps, pos=tags)
        sentences_to_visualize.append(document_result)

    for line in sentences_to_visualize:  # render all sentences through displaCy
        html_strings.append(displacy.render(line, style="dep",
                                            options={"compact": True, "word_spacing": 30, "distance": 100,
                                                     "arrow_spacing": 20}, jupyter=False))
    return html_strings


def find_nth(haystack, needle, n):
    """
    Returns the starting index of the nth occurrence of the substring 'needle' in the string 'haystack'.
    """
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


def round_base(num, base=10):
    """
    Rounding a number to its nearest multiple of the base. round_base(49.2, base=50) = 50.
    """
    return base * round(num/base)


def process_sentence_html(orig_html, semgrex_sentence):
    """
    Takes a semgrex sentence object and modifies the HTML of the original sentence's deprel visualization,
    highlighting words involved in the search queries and adding the label of the word inside of the semgrex match.

    Returns the modified html string of the sentence's deprel visualization.
    """
    tracker = {}  # keep track of which words have multiple labels
    DEFAULT_TSPAN_COUNT = 2  # the original displacy html assigns two <tspan> objects per <text> object
    CLOSING_TSPAN_LEN = 8  # </tspan> is 8 chars long
    colors = ['red', 'blue', 'purple', 'orange', 'brown', 'green']  # Colors to highlight with
    css_bolded_class = "<style> .bolded{font-weight: bold;} </style>\n"
    found_index = orig_html.find("\n")  # returns index where the opening <svg> ends
    # insert the new style class into html string
    orig_html = orig_html[: found_index + 1] + css_bolded_class + orig_html[found_index + 1:]

    # Add color to words in the match, bold words in the match
    for query in semgrex_sentence.result:
        for i, match in enumerate(query.match):
            color = colors[i]
            paired_dy = 2
            for node in match.node:
                name, match_index = node.name, node.matchIndex
                # edit existing <tspan> to change color and bold the text
                start = find_nth(orig_html, "<text", match_index)  # finds start of svg <text> of interest
                if match_index not in tracker:  # if we've already bolded and colored, keep the first color
                    tspan_start = orig_html.find("<tspan",
                                                 start)  # finds start of the first svg <tspan> inside of the <text>
                    tspan_end = orig_html.find("</tspan>", start)  # finds start of the end of the above <tspan>
                    tspan_substr = orig_html[tspan_start: tspan_end + CLOSING_TSPAN_LEN + 1] + "\n"
                    # color words in the hit and bold words in the hit
                    edited_tspan = tspan_substr.replace('class="displacy-word"', 'class="bolded"').replace(
                        'fill="currentColor"', f'fill="{color}"')
                    # insert edited <tspan> object into html string
                    orig_html = orig_html[: tspan_start] + edited_tspan + orig_html[tspan_end + CLOSING_TSPAN_LEN + 2:]
                    tracker[match_index] = DEFAULT_TSPAN_COUNT

                # next, we have to insert the new <tspan> object for the label
                # Copy old <tspan> to copy formatting when creating new <tspan> later
                prev_tspan_start = find_nth(orig_html[start:], "<tspan",
                                            tracker[match_index] - 1) + start  # find the previous <tspan> start index
                prev_tspan_end = find_nth(orig_html[start:], "</tspan>",
                                          tracker[match_index] - 1) + start  # find the prev </tspan> start index
                prev_tspan = orig_html[prev_tspan_start: prev_tspan_end + CLOSING_TSPAN_LEN + 1]

                # Find spot to insert new tspan
                closing_tspan_start = find_nth(orig_html[start:], "</tspan>", tracker[match_index]) + start
                up_to_new_tspan = orig_html[: closing_tspan_start + CLOSING_TSPAN_LEN + 1]
                rest_need_add_newline = orig_html[closing_tspan_start + CLOSING_TSPAN_LEN + 1:]

                # Calculate proper x value in svg
                x_value_start = prev_tspan.find('x="')
                x_value_end = prev_tspan[x_value_start + 3:].find('"') + 3  # 3 is the length of the 'x="' substring
                x_value = prev_tspan[x_value_start + 3: x_value_end + x_value_start]

                # Calculate proper y value in svg
                DEFAULT_DY_VAL, dy = 2, 2
                if paired_dy != DEFAULT_DY_VAL and node == match.node[
                    1]:  # we're on the second node and need to adjust height to match the paired node
                    dy = paired_dy
                if node == match.node[0]:
                    paired_node_level = 2
                    if match.node[1].matchIndex in tracker:  # check if we need to adjust heights of labels
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
                orig_html = up_to_new_tspan + new_tspan + rest_need_add_newline
                tracker[match_index] += 1
    return orig_html


def render_html_strings(edited_html_strings):
    """
    Renders the HTML to make the edits visible
    """
    for html_string in edited_html_strings:
        display(HTML(html_string))


def visualize_search_doc(doc, semgrex_queries, lang_code, start_match=0, end_match=10):
    """
    Visualizes the semgrex results of running semgrex search on a stanza doc object with the given list of
    semgrex queries. Returns a list of the edited HTML strings from the doc. Each element in the list represents
    the HTML to render one of the sentences in the document.

    'lang_code' is the two-letter language abbreviation for the language that the stanza doc object is written in.


    'start_match' and 'end_match' determine which matches to visualize. Works similar to splices, so that
    start_match=0 and end_match=10 will display the first 10 semgrex matches.
    """
    matches_count = 0  # Limits number of visualizations
    with Semgrex(classpath="$CLASSPATH") as sem:
        edited_html_strings = []
        semgrex_results = sem.process(doc, *semgrex_queries)
        # one html string for each sentence
        unedited_html_strings = get_sentences_html(doc, lang_code)
        for i in range(len(unedited_html_strings)):

            if matches_count >= end_match:  # we've collected enough matches, stop early
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
                    edited_string = process_sentence_html(unedited_html_strings[i], semgrex_results.result[i])
                    edited_string = adjust_dep_arrows(edited_string)
                    edited_html_strings.append(edited_string)
                matches_count += 1

        render_html_strings(edited_html_strings)
    return edited_html_strings


def visualize_search_str(text, semgrex_queries, lang_code):
    """
    Visualizes the deprel of the semgrex results from running semgrex search on a string with the given list of
    semgrex queries. Returns a list of the edited HTML strings. Each element in the list represents
    the HTML to render one of the sentences in the document.

    Internally, this function converts the string into a stanza doc object before processing the doc object.

    'lang_code' is the two-letter language abbreviation for the language that the stanza doc object is written in.
    """
    nlp = stanza.Pipeline(lang_code, processors="tokenize, pos, lemma, depparse")
    doc = nlp(text)
    return visualize_search_doc(doc, semgrex_queries, lang_code)


def adjust_dep_arrows(raw_html):
    """
    The default spaCy dependency visualization has misaligned arrows.
    We fix arrows by aligning arrow ends and bodies to the word that they are directed to. If a word has an
    arrowhead that is pointing not directly on the word's center, align the arrowhead to match the center of the word.

    returns the edited html with fixed arrow placement
    """
    HTML_ARROW_BEGINNING = '<g class="displacy-arrow">'
    HTML_ARROW_ENDING = "</g>"
    HTML_ARROW_ENDING_LEN = 6   # there are 2 newline chars after the arrow ending
    arrows_start_idx = find_nth(haystack=raw_html, needle='<g class="displacy-arrow">', n=1)
    words_html, arrows_html = raw_html[: arrows_start_idx], raw_html[arrows_start_idx:]  # separate html for words and arrows
    final_html = words_html  # continually concatenate to this after processing each arrow
    arrow_number = 1  # which arrow we're editing (1-indexed)
    start_idx, end_of_class_idx = find_nth(haystack=arrows_html, needle=HTML_ARROW_BEGINNING, n=arrow_number), find_nth(arrows_html, HTML_ARROW_ENDING, arrow_number)
    while start_idx != -1:  # edit every arrow
        arrow_section = arrows_html[start_idx: end_of_class_idx + HTML_ARROW_ENDING_LEN]  # slice a single svg arrow object
        if arrow_section[-1] == "<":   # this is the last arrow in the HTML, don't cut the splice early
            arrow_section = arrows_html[start_idx:]
        edited_arrow_section = edit_dep_arrow(arrow_section)

        final_html = final_html + edited_arrow_section  # continually update html with new arrow html until done

        # Prepare for next iteration
        arrow_number += 1
        start_idx = find_nth(arrows_html, '<g class="displacy-arrow">', n=arrow_number)
        end_of_class_idx = find_nth(arrows_html, "</g>", arrow_number)
    return final_html


def edit_dep_arrow(arrow_html):
    """
    The formatting of a displacy arrow in svg is the following:
    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-c628889ffbf343e3848193a08606f10a-0-0" stroke-width="2px" d="M70,352.0 C70,177.0 390.0,177.0 390.0,352.0" fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-c628889ffbf343e3848193a08606f10a-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">csubj</textPath>
        </text>
        <path class="displacy-arrowhead" d="M70,354.0 L62,342.0 78,342.0" fill="currentColor"/>
    </g>

    We edit the 'd = ...' parts of the <path class ...> section to fix the arrow direction and length

    returns the arrow_html with distances fixed
    """
    WORD_SPACING = 50   # words start at x=50 and are separated by 100s so their x values are multiples of 50
    M_OFFSET = 4  # length of 'd="M' that we search for to extract the number from d="M70, for instance
    ARROW_PIXEL_SIZE = 4
    first_d_idx, second_d_idx = find_nth(arrow_html, 'd="M', 1), find_nth(arrow_html, 'd="M', 2)  # find where d="M starts
    first_d_cutoff, second_d_cutoff = arrow_html.find(",", first_d_idx), arrow_html.find(",", second_d_idx)  # isolate the number after 'M' e.g. 'M70'
    # gives svg x values of arrow body starting position and arrowhead position
    arrow_position, arrowhead_position = float(arrow_html[first_d_idx + M_OFFSET: first_d_cutoff]), float(arrow_html[second_d_idx + M_OFFSET: second_d_cutoff])
    # gives starting index of where 'fill="none"' or 'fill="currentColor"' begin, reference points to end the d= section
    first_fill_start_idx, second_fill_start_idx = find_nth(arrow_html, "fill", n=1), find_nth(arrow_html, "fill", n=3)

    # isolate the d= ... section to edit
    first_d, second_d = arrow_html[first_d_idx: first_fill_start_idx], arrow_html[second_d_idx: second_fill_start_idx]
    first_d_split, second_d_split = first_d.split(","), second_d.split(",")

    if arrow_position == arrowhead_position:  # This arrow is incoming onto the word, center the arrow/head to word center
        corrected_arrow_pos = corrected_arrowhead_pos = round_base(arrow_position, base=WORD_SPACING)

        # edit first_d  -- arrow body
        second_term = first_d_split[1].split(" ")[0] + " " + str(corrected_arrow_pos)
        first_d = 'd="M' + str(corrected_arrow_pos) + "," + second_term + "," + ",".join(first_d_split[2:])

        # edit second_d  -- arrowhead
        second_term = second_d_split[1].split(" ")[0] + " L" + str(corrected_arrowhead_pos - ARROW_PIXEL_SIZE)
        third_term = second_d_split[2].split(" ")[0] + " " + str(corrected_arrowhead_pos + ARROW_PIXEL_SIZE)
        second_d = 'd="M' + str(corrected_arrowhead_pos) + "," + second_term + "," + third_term + "," + ",".join(second_d_split[3:])
    else:  # This arrow is outgoing to another word, center the arrow/head to that word's center
        corrected_arrowhead_pos = round_base(arrowhead_position, base=WORD_SPACING)

        # edit first_d -- arrow body
        third_term = first_d_split[2].split(" ")[0] + " " + str(corrected_arrowhead_pos)
        fourth_term = first_d_split[3].split(" ")[0] + " " + str(corrected_arrowhead_pos)
        terms = [first_d_split[0], first_d_split[1], third_term, fourth_term] + first_d_split[4:]
        first_d = ",".join(terms)

        # edit second_d -- arrow head
        first_term = f'd="M{corrected_arrowhead_pos}'
        second_term = second_d_split[1].split(" ")[0] + " L" + str(corrected_arrowhead_pos - ARROW_PIXEL_SIZE)
        third_term = second_d_split[2].split(" ")[0] + " " + str(corrected_arrowhead_pos + ARROW_PIXEL_SIZE)
        terms = [first_term, second_term, third_term] + second_d_split[3:]
        second_d = ",".join(terms)
    # rebuild and return html
    return arrow_html[:first_d_idx] + first_d + " " + arrow_html[first_fill_start_idx:second_d_idx] + second_d + " " + arrow_html[second_fill_start_idx:]


def main():
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

    # doc = nlp("This a dummy sentence. Banning opal removed all artifact decks from the meta.  I miss playing lantern. This is a dummy sentence.")
    doc = nlp("Banning opal removed artifact decks from the meta. Banning tennis resulted in players banning people.")
    # A single result .result[i].result[j] is a list of matches for sentence i on semgrex query j.
    queries = ["{pos:NN}=object <obl {}=action",
               "{cpos:NOUN}=thing <obj {cpos:VERB}=action"]
    res = visualize_search_doc(doc, queries, "en")
    print(res[0])  # see the first sentence's deprel visualization HTML
    print("---------------------------------------")
    print(res[1])  # second sentence's deprel visualization HTML
    return


if __name__ == '__main__':
    main()
