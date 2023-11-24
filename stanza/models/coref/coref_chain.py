"""
Coref chain suitable for attaching to a Document after coref processing
"""

from collections import namedtuple

CorefMention = namedtuple('CorefMention', ['sentence', 'start_word', 'end_word'])
CorefChain = namedtuple('CorefChain', ['mentions', 'representative', 'representative_text'])
