import stanza

def load_doc_from_conll_file(path: str):
    """"
    loads in a Stanza document object from a path to a CoNLL file containing annotated sentences.
    """
    return stanza.utils.conll.CoNLL.conll2doc(path)