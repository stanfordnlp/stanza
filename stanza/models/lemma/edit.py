"""
Utilities for calculating edits between word and lemma forms.
"""

EDIT_TO_ID = {'none': 0, 'identity': 1, 'lower': 2}

def get_edit_type(word, lemma):
    """ Calculate edit types. """
    if lemma == word:
        return 'identity'
    elif lemma == word.lower():
        return 'lower'
    return 'none'

def edit_word(word, pred, edit_id):
    """
    Edit a word, given edit and seq2seq predictions.
    """
    if edit_id == 1:
        return word
    elif edit_id == 2:
        return word.lower()
    elif edit_id == 0:
        return pred
    else:
        raise Exception("Unrecognized edit ID: {}".format(edit_id))

