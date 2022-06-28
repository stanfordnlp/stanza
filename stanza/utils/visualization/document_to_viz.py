import stanza
import spacy
from spacy import displacy
from spacy.tokens import Doc


def visualize_doc(doc):
    """
    Takes in a Document and visualizes it using displacy. The document must be from the stanza pipeline.
    Works for English inputs.
    """
    nlp = spacy.load("en_core_web_sm")
    words = []
    lemmas = []
    heads = []
    deps = []
    tags = []
    for word in doc.sentences[0].words:
        print(word.text, word.lemma, word.upos, word.head, word.deprel)
        print(word)
        words.append(word.text)
        lemmas.append(word.lemma)
        deps.append(word.deprel)
        tags.append(word.upos)
        if word.head == 0:   # spaCy head indexes are formatted differently than that of Stanza's
            heads.append(word.id - 1)
        else:
            heads.append(word.head - 1)
    # print(words, lemmas, heads, deps, tags)
    document_result = Doc(nlp.vocab, words=words, lemmas=lemmas, heads=heads, deps=deps, pos=tags)
    displacy.serve(document_result, style="dep")


def visualize_str(text):
    """
    Takes a string and visualizes it using displacy. The string is processed using the stanza pipeline and
    its dependencies are formatted into a spaCy doc object for easy visualization. Accepts English text.
    """
    pipe = stanza.Pipeline("en")
    doc = pipe(text)
    visualize_doc(doc)


def main():
    visualize_str("A power outage crippled Stanford for 3 days.")
    pipe = stanza.Pipeline("en")
    doc = pipe("A power outage crippled Stanford for 3 days.")
    print("VISUALIZING DOCUMENT:\n")
    visualize_doc(doc)


main()
