import stanza
import spacy
from spacy import displacy
from spacy.tokens import Doc


def visualize_doc(doc, reverse_order=False):
    """
    Takes in a Document and visualizes it using displacy. The document must be from the stanza pipeline.
    Works for English inputs. The reverse_order parameter can be set as True to flip the display of the
    words for languages such as Arabic, which are read from right-to-left.
    """
    visualization_options = {"compact": True, "bg": "#09a3d5", "color": "white", "distance": 80,
                             "font": "Source Sans Pro"}
    nlp = spacy.load("en_core_web_sm")
    sentences_to_visualize = []
    for sentence in doc.sentences:
        words, lemmas, heads, deps, tags = [], [], [], [], []
        if reverse_order:  # order of words displayed is reversed, dependency arcs remain intact
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
        else:
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

        for line in sentences_to_visualize:  # render using displaCy
            displacy.render(line, style="dep", options=visualization_options)


def visualize_str(text, pipeline, reverse=False):
    """
    Takes a string and visualizes it using displacy. The string is processed using the stanza pipeline and
    its dependencies are formatted into a spaCy doc object for easy visualization. Accepts valid stanza (UD)
    pipelines as the pipeline argument.
    """
    pipe = stanza.Pipeline(pipeline)
    doc = pipe(text)
    visualize_doc(doc, reverse)


def main():
    print("PRINTING ARABIC DOCUMENTS")
    # example sentences
    visualize_str('برلين ترفض حصول شركة اميركية على رخصة تصنيع دبابة "ليوبارد" الالمانية', "ar", reverse=True)
    visualize_str("هل بإمكاني مساعدتك؟", "ar", reverse=True)
    visualize_str("أراك في مابعد", "ar", reverse=True)
    visualize_str("لحظة من فضلك", "ar", reverse=True)


main()
