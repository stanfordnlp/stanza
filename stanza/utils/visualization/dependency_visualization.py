from stanza.models.common.constant import is_right_to_left
import stanza
import spacy
from spacy import displacy
from spacy.tokens import Doc


def visualize_doc(doc, language):
    """
    Takes in a Document and visualizes it using displacy.

    The document to visualize must be from the stanza pipeline.

    right-to-left languages such as Arabic are displayed right-to-left based on the language code
    """
    visualization_options = {"compact": True, "bg": "#09a3d5", "color": "white", "distance": 90,
                             "font": "Source Sans Pro", "arrow_spacing": 25}
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
        else:   # left to right rendering
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
        # If this program is NOT being run in a Jupyter notebook, replace displacy.render with displacy.serve
        # and the visualization will be hosted locally, link being provided in the program output.
        displacy.render(line, style="dep", options=visualization_options)


def visualize_str(text, pipeline_code, pipe):
    """
    Takes a string and visualizes it using displacy.

    The string is processed using the stanza pipeline and its
    dependencies are formatted into a spaCy doc object for easy
    visualization. Accepts valid stanza (UD) pipelines as the pipeline
    argument. Must supply the stanza pipeline code (the two-letter
    abbreviation of the language, such as 'en' for English. Must also
    supply the stanza pipeline object as the third argument.
    """
    doc = pipe(text)
    visualize_doc(doc, pipeline_code)


def visualize_docs(docs, lang_code):
    """
    Takes in a list of Stanza document objects and a language code (ex: 'en' for English) and visualizes the
    dependency relationships within each document.

    This function uses spaCy visualizations. See the visualize_doc function for more details.
    """
    for doc in docs:
        visualize_doc(doc, lang_code)


def visualize_strings(texts, lang_code):
    """
    Takes a language code (ex: 'en' for English) and a list of strings to process and visualizes the
    dependency relationships in each text.

    This function loads the Stanza pipeline for the given language and uses it to visualize all of the strings provided.
    """
    pipe = stanza.Pipeline(lang_code, processors="tokenize,pos,lemma,depparse")
    for text in texts:
        visualize_str(text, lang_code, pipe)


def main():
    ar_strings = ['برلين ترفض حصول شركة اميركية على رخصة تصنيع دبابة "ليوبارد" الالمانية', "هل بإمكاني مساعدتك؟",
               "أراك في مابعد", "لحظة من فضلك"]
    en_strings = ["This is a sentence.",
                  "Barack Obama was born in Hawaii. He was elected President of the United States in 2008."]
    zh_strings = ["中国是一个很有意思的国家。"]
    # Testing with right to left language
    visualize_strings(ar_strings, "ar")
    # Testing with left to right languages
    visualize_strings(en_strings, "en")
    visualize_strings(zh_strings, "zh")

if __name__ == '__main__':
    main()
