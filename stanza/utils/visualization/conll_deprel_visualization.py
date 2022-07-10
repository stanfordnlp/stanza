from stanza.models.common.constant import is_right_to_left
import spacy
from spacy import displacy
from spacy.tokens import Doc
from stanza.utils import conll
from stanza.utils.visualization import document_to_viz as viz


def conll_to_visual(conll_file, pipeline, sent_count=10, display_all=False):
    """
    Takes in a conll file and visualizes it by converting the conll file to a Stanza Document object
    and visualizing it with the visualize_doc method. Input should be a proper conll file. The pipeline
    for the conll file to be processed in must be provided as well. Optionally, the sent_count argument
    can be tweaked to display a different amount of sentences. To display all of the sentences in a
    conll file, the display_all argument can optionally be set to True. BEWARE: setting this argument
    for a large conll file may result in too many renderings, resulting in a crash.
    """
    # convert conll file to doc
    doc = conll.CoNLL.conll2doc(conll_file)

    if display_all:
        viz.visualize_doc(conll.CoNLL.conll2doc(conll_file), pipeline)
    else:  # visualize a given number of sentences
        visualization_options = {"compact": True, "bg": "#09a3d5", "color": "white", "distance": 100,
                                 "font": "Source Sans Pro", "offset_x": 30,
                                 "arrow_spacing": 20}  # see spaCy visualization settings doc for more options
        nlp = spacy.blank("en")  # Must install the latest version of spaCy's en_core_web_sm before running
        sentences_to_visualize, rtl = [], is_right_to_left(pipeline)

        for i in range(sent_count):
            sentence = doc.sentences[i]
            words, lemmas, heads, deps, tags = [], [], [], [], []
            sentence_words = sentence.words
            if rtl:
                sentence_words = reversed(sentence.words)
                sent_len = len(sentence.words)
            for word in sentence_words:
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                tags.append(word.upos)
                if rtl and word.head == 0:  # word heads are off-by-1 in spaCy doc inits compared to Stanza
                    heads.append(sent_len - word.id)
                elif rtl and word.head != 0:
                    heads.append(sent_len - word.head)
                elif not rtl and word.head == 0:
                    heads.append(word.id - 1)
                elif not rtl and word.head != 0:
                    heads.append(word.head - 1)

            document_result = Doc(nlp.vocab, words=words, lemmas=lemmas, heads=heads, deps=deps, pos=tags)
            sentences_to_visualize.append(document_result)

        for line in sentences_to_visualize:  # render all sentences through displaCy
            displacy.render(line, style="dep", options=visualization_options)


def main():
    file = "C:\\Users\\Alex\\stanza\\stanza\\utils\\visualization\\en_ewt-ud-test.conllu.txt"
    conll_to_visual(file, "en", sent_count=2)
    # conll_to_visual(file, "en")
    # conll_to_visual(file, "en", sent_count=20)
    # conll_to_visual(file, "en", display_all=True)
    jp_file = "C:\\Users\\Alex\\stanza\\stanza\\utils\\visualization\\ja_gsd-ud-test.conllu.txt"
    ar_file = "C:\\Users\\Alex\\stanza\\stanza\\utils\\visualization\\ar_padt-ud-test.conllu.txt"
    conll_to_visual(jp_file, "ja")
    conll_to_visual(ar_file, "ar")


main()