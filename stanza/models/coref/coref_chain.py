"""
Coref chain suitable for attaching to a Document after coref processing
"""

# by not using namedtuple, we can use this object as output from the json module
# in the doc class as long as we wrap the encoder to print these out in dict() form
# CorefMention = namedtuple('CorefMention', ['sentence', 'start_word', 'end_word'])
class CorefMention:
    def __init__(self, sentence, start_word, end_word):
        self.sentence = sentence
        self.start_word = start_word
        self.end_word = end_word

class CorefChain:
    def __init__(self, index, mentions, representative_text, representative_index):
        self.index = index
        self.mentions = mentions
        self.representative_text = representative_text
        self.representative_index = representative_index

class CorefAttachment:
    def __init__(self, chain, is_start, is_end, is_representative):
        self.chain = chain
        self.is_start = is_start
        self.is_end = is_end
        self.is_representative = is_representative

    def to_json(self):
        j = {
            "index": self.chain.index,
            "representative_text": self.chain.representative_text
        }
        if self.is_start:
            j['is_start'] = True
        if self.is_end:
            j['is_end'] = True
        if self.is_representative:
            j['is_representative'] = True
        return j
