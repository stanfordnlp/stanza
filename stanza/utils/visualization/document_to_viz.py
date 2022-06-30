from stanza.models.common.constant import is_right_to_left
import stanza
import spacy
from spacy import displacy
from spacy.tokens import Doc


def visualize_doc(doc, pipeline):
    """
    Takes in a Document and visualizes it using displacy. The document must be from the stanza pipeline.
    Works for English inputs. The reverse_order parameter can be set as True to flip the display of the
    words for languages such as Arabic, which are read from right-to-left.
    """
    visualization_options = {"compact": True, "bg": "#09a3d5", "color": "white", "distance": 80,
                             "font": "Source Sans Pro"}
    nlp = spacy.load("en_core_web_sm")   # Must install the latest version of spaCy's en_core_web_sm before running
    sentences_to_visualize = []
    for sentence in doc.sentences:
        words, lemmas, heads, deps, tags = [], [], [], [], []
        if is_right_to_left(pipeline):  # order of words displayed is reversed, dependency arcs remain intact
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
        else:   # standard left to right rendering
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
            displacy.render(line, style="dep", options=visualization_options)


def visualize_str(text, pipeline):
    """
    Takes a string and visualizes it using displacy. The string is processed using the stanza pipeline and
    its dependencies are formatted into a spaCy doc object for easy visualization. Accepts valid stanza (UD)
    pipelines as the pipeline argument.
    """
    pipe = stanza.Pipeline(pipeline)
    doc = pipe(text)
    visualize_doc(doc, pipeline)


def main():
    print("PRINTING ARABIC DOCUMENTS")
    # example sentences in right to left language
    visualize_str('برلين ترفض حصول شركة اميركية على رخصة تصنيع دبابة "ليوبارد" الالمانية', "ar")
    visualize_str("هل بإمكاني مساعدتك؟", "ar")
    visualize_str("أراك في مابعد", "ar")
    visualize_str("لحظة من فضلك", "ar")
    # example sentences in left to right language
    visualize_str("This is a sentence.", "en")
    visualize_str("中国是一个很有意思的国家。", "zh")


main()
