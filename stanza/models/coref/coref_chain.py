"""
Coref chain suitable for attaching to a Document after coref processing
"""

from collections import namedtuple

# TODO: it would probably be nicer to include the text, at least of the representative mention
# TODO: maybe also add back pointers from the words to the coref chains?
CorefMention = namedtuple('CorefMention', ['sentence', 'start_word', 'end_word'])
CorefChain = namedtuple('CorefChain', ['mentions', 'representative'])
