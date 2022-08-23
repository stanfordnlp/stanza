import stanza
from stanza.server.semgrex import Semgrex
from stanza.models.common.constant import is_right_to_left
import spacy
from spacy import displacy
from spacy.tokens import Doc
from IPython.core.display import display, HTML



def get_sentences_html(doc, language):
    """
    Returns a list of the HTML strings of the dependency visualizations of a given stanza doc object.

    First converts the stanza doc object to a spacy doc object and uses displacy to generate an HTML
    string for each sentence of the doc object.
    """
    html_strings = []

    # blank model - we don't use any of the model features, just the viz
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
        html_strings.append(displacy.render(line, style="dep", jupyter=False))
    return html_strings


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start


def process_sentence_html(orig_html, semgrex_sentence):
        """
        Takes the semgrex sentence object and modifies the HTML of the original sentence, highlighting
        words involved in the search queries and adding the label of the word inside of the semgrex match.
        """
        tracker = {}  # keep track of which words have multiple labels
        DEFAULT_TSPAN_COUNT = 2   # the original displacy html assigns two <tspan> objects per <text> object
        CLOSING_TSPAN_LEN = 8   # </tspan> is 8 chars long
        colors = ['red', 'blue', 'purple', 'orange', 'brown', 'green']
        bolded_class = "<style> .bolded{font-weight: bold;} </style>\n"
        found_index = orig_html.find("\n")   # returns index where the <svg> ends
        # insert the new style class into html string
        orig_html = orig_html[: found_index + 1] + bolded_class + orig_html[found_index + 1: ]

        # Color words in the match, bold words in the match
        for query in semgrex_sentence.result:
            for i, match in enumerate(query.match):
                color = colors[i]
                paired_dy = 2
                for node in match.node:
                    name, match_index = node.name, node.matchIndex
                    # edit existing <tspan>
                    start = find_nth(orig_html, "<text", match_index)  # finds start of svg <text> of interest
                    if match_index not in tracker:  # if we've already bolded and colored, keep the first color
                        tspan_start = orig_html.find("<tspan", start)   # finds start of the first svg <tspan> inside of the <text>
                        tspan_end = orig_html.find("</tspan>", start)  # finds start of the end of the above <tspan>
                        tspan_substr = orig_html[tspan_start: tspan_end + CLOSING_TSPAN_LEN + 1] + "\n"
                        # color words in the hit and bold words in the hit
                        edited_tspan = tspan_substr.replace('class="displacy-word"', 'class="bolded"').replace('fill="currentColor"', f'fill="{color}"')
                        # insert edited <tspan> object into html string
                        orig_html = orig_html[: tspan_start] + edited_tspan + orig_html[tspan_end + CLOSING_TSPAN_LEN + 2: ]
                        tracker[match_index] = DEFAULT_TSPAN_COUNT

                    # next, we have to insert the new <tspan> object for the label
                    # Copy old <tspan> to copy formatting when creating new <tspan> later
                    prev_tspan_start = find_nth(orig_html[start:], "<tspan", tracker[match_index] - 1) + start  # find the previous <tspan> start index
                    prev_tspan_end = find_nth(orig_html[start: ], "</tspan>", tracker[match_index]- 1) + start  # find the prev </tspan> start index
                    prev_tspan = orig_html[prev_tspan_start: prev_tspan_end + CLOSING_TSPAN_LEN + 1]

                    # Find spot to insert new tspan
                    closing_tspan_start = find_nth(orig_html[start: ], "</tspan>", tracker[match_index]) + start
                    up_to_new_tspan = orig_html[: closing_tspan_start + CLOSING_TSPAN_LEN + 1]
                    rest_need_add_newline = orig_html[closing_tspan_start + CLOSING_TSPAN_LEN + 1: ]

                    # Calculate proper x value
                    x_value_start = prev_tspan.find('x="')
                    x_value_end = prev_tspan[x_value_start + 3:].find('"') + 3  # 3 is the length of the 'x="' substring
                    x_value = prev_tspan[x_value_start + 3: x_value_end + x_value_start]

                    # Calculate proper y value
                    DEFAULT_DY_VAL, dy = 2, 2
                    if paired_dy != DEFAULT_DY_VAL and node == match.node[1]:
                        dy = paired_dy
                    if node == match.node[0]:
                        paired_node_level = 2
                        if match.node[1].matchIndex in tracker:
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


# Once HTMLs are configured:
def render_html_strings(edited_html_strings):
    for html_string in edited_html_strings:
        display(HTML(html_string))
    return


def visualize_search_doc(doc, semgrex_queries, lang_code):
    # nlp = stanza.Pipeline(lang_code, processors="tokenize,pos,lemma,depparse")

    # doc = nlp("Banning opal removed all artifact decks from the meta.  I miss playing lantern.")

    # A single result .result[i].result[j] is a list of matches for sentence i on semgrex query j.

    with Semgrex(classpath="$CLASSPATH") as sem:
        # semgrex_results = sem.process(doc,
        #                               "{pos:NN}=object <obl {}=action",
        #                               "{cpos:NOUN}=thing <obj {cpos:VERB}=action")
        edited_html_strings = []
        semgrex_results = sem.process(doc, *semgrex_queries)
        # one html string for each sentence
        html_strings = get_sentences_html(doc, lang_code)
        for i in range(len(html_strings)):
            edited_string = process_sentence_html(html_strings[i], semgrex_results.result[i])
            edited_html_strings.append(edited_string)
        render_html_strings(edited_html_strings)
    return edited_html_strings


def visualize_search_str(text, semgrex_queries, lang_code):
    nlp = stanza.Pipeline(lang_code, processors="tokenize, pos, lemma, depparse")
    doc = nlp(text)
    visualize_search_doc(doc, semgrex_queries, lang_code)


def main():
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

    doc = nlp("Banning opal removed all artifact decks from the meta.  I miss playing lantern.")

    # A single result .result[i].result[j] is a list of matches for sentence i on semgrex query j.
    queries = ["{pos:NN}=object <obl {}=action",
                                      "{cpos:NOUN}=thing <obj {cpos:VERB}=action"]
    res = visualize_search_doc(doc, queries, "en")
    print(res[0])
    print("---------------------------------------")
    print(res[1])

    with Semgrex(classpath="$CLASSPATH") as sem:
        semgrex_results = sem.process(doc,
                                      "{pos:NN}=object <obl {}=action",
                                      "{cpos:NOUN}=thing <obj {cpos:VERB}=action")
        print(semgrex_results.result[1])
        print(semgrex_results.result[1].result[1])
        print("-------------------------------------")
        print(semgrex_results.result[1].result[1].match[0].node[0])
        print(semgrex_results.result[1].result[1].match[0].node[0].name)
        print(semgrex_results.result[1].result[1].match[0].node[0].matchIndex)

        print(semgrex_results.result[1].result[1].match[0].node[1].name)
        print(semgrex_results.result[1].result[1].match[0].node[1].matchIndex)

        html = '''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="591e75870ca849b08a7784f20b0d6c69-0" class="displacy" width="750" height="224.5" direction="ltr" style="max-width: none; height: 224.5px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr">
    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="134.5">
        <tspan class="displacy-word" fill="currentColor" x="50">I</tspan>
        <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">PRON</tspan>
    </text>
    
    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="134.5">
        <tspan class="displacy-word" fill="currentColor" x="225">miss</tspan>
        <tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">VERB</tspan>
    </text>
    
    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="134.5">
        <tspan class="bolded" fill="red" x="400">playing</tspan>
        <tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">VERB</tspan>
        <tspan class="displacy-word" dy="2em" fill="currentColor" x="400">Action</tspan>
    </text>
    
    <text class="displacy-token" fill="currentColor" text-anchor="middle" y="134.5">
        <tspan class="bolded" fill="red" x="575">lantern .</tspan>
        <tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">NOUN</tspan>
        <tspan class="displacy-word" dy="2em" fill="currentColor" x="575">Thing</tspan>
    </text>
    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-092faa2460734465b35453bd9e020480-0-0" stroke-width="2px" d="M70,89.5 C70,2.0 225.0,2.0 225.0,89.5" fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-092faa2460734465b35453bd9e020480-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
        </text>
        <path class="displacy-arrowhead" d="M70,91.5 L62,79.5 78,79.5" fill="currentColor"/>
    </g>
    
    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-092faa2460734465b35453bd9e020480-0-1" stroke-width="2px" d="M245,89.5 C245,2.0 400.0,2.0 400.0,89.5" fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-092faa2460734465b35453bd9e020480-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">xcomp</textPath>
        </text>
        <path class="displacy-arrowhead" d="M400.0,91.5 L408.0,79.5 392.0,79.5" fill="currentColor"/>
    </g>
    
    <g class="displacy-arrow">
        <path class="displacy-arc" id="arrow-092faa2460734465b35453bd9e020480-0-2" stroke-width="2px" d="M420,89.5 C420,2.0 575.0,2.0 575.0,89.5" fill="none" stroke="currentColor"/>
        <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
            <textPath xlink:href="#arrow-092faa2460734465b35453bd9e020480-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">obj</textPath>
        </text>
        <path class="displacy-arrowhead" d="M575.0,91.5 L583.0,79.5 567.0,79.5" fill="currentColor"/>
    </g>
    </svg>
    '''
        process_sentence_html(html, semgrex_results.result[1])
    return


if __name__ == '__main__':
    main()
