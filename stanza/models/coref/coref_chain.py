"""
Data classes for storing coreference resolution results attached to a Document.

After running the coref pipeline processor, a Document's `.coref` attribute holds
a list of CorefChain objects. Each CorefChain groups all the mentions (spans of
text) that refer to the same entity. Individual words/tokens also receive
CorefAttachment objects (via their `.coref` attribute) that link back to the
chain(s) they participate in.

Typical usage after running a coref-enabled pipeline::

    doc = nlp("Barack Obama was born in Hawaii. He later became president.")
    for chain in doc.coref:
        print(chain.representative_text, "->", [(m.sentence, m.start_word, m.end_word) for m in chain.mentions])

Barack Obama -> [(0, 0, 2), (1, 0, 1)]
Hawaii -> [(0, 5, 6)]
president -> [(1, 3, 4)]

See also:
    stanza.pipeline.coref_processor  -- builds CorefMention and CorefChain objects
    stanza.models.common.doc         -- attaches CorefAttachment to Word objects
"""

# by not using namedtuple, we can use this object as output from the json module
# in the doc class as long as we wrap the encoder to print these out in dict() form
# CorefMention = namedtuple('CorefMention', ['sentence', 'start_word', 'end_word'])
class CorefMention:
    """
    A single mention of an entity: a contiguous span of words within one sentence.

    Mentions are always confined to a single sentence (the coref model never
    predicts cross-sentence spans). Indices follow the same 0-based word
    indexing used internally by the coref model, i.e. the first word of a
    sentence is index 0. Note that this differs from the 1-based `.id` values
    on Word/Token objects in the rest of Stanza.

    For zero anaphora (dropped pronouns), both start_word and end_word are set
    to the same tuple value of the form (word_id, empty_word_index), mirroring
    the enhanced UD convention for empty nodes.

    Attributes:
        sentence (int): 0-based index of the sentence within the document that
            contains this mention.
        start_word (int | tuple): 0-based index of the first word of the mention
            within its sentence. For zero-anaphora nodes this is a tuple
            (word_id, empty_node_index) rather than a plain int.
        end_word (int | tuple): 0-based index one past the last word of the
            mention (exclusive), so the mention covers words
            sentence.words[start_word:end_word]. For zero-anaphora nodes this
            equals start_word (a tuple).
    """
    def __init__(self, sentence, start_word, end_word):
        self.sentence = sentence
        self.start_word = start_word
        self.end_word = end_word

class CorefChain:
    """
    A coreference chain: a cluster of CorefMention objects that all refer to
    the same real-world entity.

    CorefChain objects are created by CorefProcessor and stored as a list in
    Document.coref. Chains are numbered sequentially in document order.

    The *representative* mention is heuristically chosen as the longest span in
    the cluster. Ties are broken by earliest position in the document, and if
    the POS processor has been run, secondarily by the mention containing the
    most PROPN (proper noun) tokens.

    Attributes:
        index (int): 0-based position of this chain in the document's
            ``doc.coref`` list.
        mentions (list[CorefMention]): All mentions belonging to this chain,
            in the order they were found by the model (roughly document order
            after sorting).
        representative_text (str): The surface text of the representative
            mention — the longest (most informative) span in the cluster.
            Set to ``"_"`` for clusters whose only mentions are zero-anaphora
            nodes (which have no surface text).
        representative_index (int | None): Index into ``mentions`` of the
            representative mention, or ``None`` if no suitable span was found
            (e.g. all mentions are zero nodes).
    """
    def __init__(self, index, mentions, representative_text, representative_index):
        self.index = index
        self.mentions = mentions
        self.representative_text = representative_text
        self.representative_index = representative_index

class CorefAttachment:
    """
    A back-reference from a Word to the CorefChain it participates in.

    Each Word that is part of a coreference mention receives a list of
    CorefAttachment objects in its ``.coref`` attribute (one per chain the word
    belongs to — a word can be part of multiple overlapping mentions in
    different chains). CorefAttachment objects are created by
    ``Document._set_up_coref_attachments()`` in ``stanza/models/common/doc.py``.

    The flags ``is_start`` and ``is_end`` indicate whether this particular word
    is the first or last word of its mention within the chain (useful for
    reconstructing mention boundaries from a flat word list without having to
    inspect the CorefChain directly). ``is_representative`` flags the single
    word that begins the representative mention of the chain.

    Attributes:
        chain (CorefChain): The coreference chain this attachment belongs to.
        is_start (bool): True if this word is the first word of the mention
            span (i.e. word index == mention.start_word).
        is_end (bool): True if this word is the last word of the mention span
            (i.e. word index == mention.end_word - 1).
        is_representative (bool): True if this word is the first word of the
            chain's representative mention.

    Note:
        ``to_json()`` intentionally omits ``is_start``, ``is_end``, and
        ``is_representative`` when they are False, to keep JSON output compact.
    """
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

    def __repr__(self):
        # not eval-usable
        flags = []
        if self.is_start:
            flags.append("start")
        if self.is_end:
            flags.append("end")
        if self.is_representative:
            flags.append("representative")
        flag_str = ", ".join(flags)
        return (
            f"CorefAttachment(chain={self.chain.index}"
            f", representative={self.chain.representative_text!r}"
            f"{', ' + flag_str if flag_str else ''})"
        )

    def __str__(self):
        return repr(self)
