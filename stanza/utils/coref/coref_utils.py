import collections

from stanza.models.common.doc import Document


def iter_coref_chains(doc):
    """
    Yield (chain_index, CorefChain) for each coreference chain in the document.
    """
    if not isinstance(doc, Document):
        raise ValueError("iter_coref_chains expects a Document")
    if not getattr(doc, "coref", None):
        return
    for chain in doc.coref:
        yield chain.index, chain


def coref_chains_as_dicts(doc):
    """
    Return coreference chains as a list of dictionaries with mention spans and representative text.
    """
    chains = []
    for chain_index, chain in iter_coref_chains(doc):
        mentions = []
        for mention in chain.mentions:
            mentions.append(
                {
                    "sentence": mention.sentence,
                    "start": mention.start_word,
                    "end": mention.end_word,
                }
            )
        chains.append(
            {
                "index": chain_index,
                "representative": chain.representative_text,
                "mentions": mentions,
            }
        )
    return chains


def _build_token_lookup(doc):
    """
    Build a flat list of (sentence_idx, word_idx) pairs for all words in the document.
    """
    mapping = []
    for sent_idx, sentence in enumerate(doc.sentences):
        for word_idx, _ in enumerate(sentence.words):
            mapping.append((sent_idx, word_idx))
    return mapping


def resolve_coref(doc, strategy="first_mention"):
    """
    Return a token list with coreferent mentions replaced by a representative string.

    The default strategy replaces all non-representative mentions in a chain
    with the chain's representative text, based on document order.
    """
    if not isinstance(doc, Document):
        raise ValueError("resolve_coref expects a Document")

    # If there is no coreference information, fall back to the raw tokens
    if not getattr(doc, "coref", None):
        return [word.text for sentence in doc.sentences for word in sentence.words]

    # Flat sequence of tokens; we use this as our working copy
    token_lookup = _build_token_lookup(doc)
    tokens = [doc.sentences[sent_idx].words[word_idx].text for (sent_idx, word_idx) in token_lookup]

    # Map from (sentence, start_word, end_word) to representative text
    mention_to_rep = {}
    for chain in doc.coref:
        if not chain.mentions:
            continue

        if strategy == "first_mention":
            # Use the earliest mention in document order as representative
            rep = min(chain.mentions, key=lambda m: (m.sentence, m.start_word))
        else:
            # Fall back to the chain's stored representative, if available
            rep = chain.mentions[chain.representative_index] if chain.representative_index is not None else chain.mentions[0]

        rep_text = chain.representative_text
        rep_key = (rep.sentence, rep.start_word, rep.end_word)

        for mention in chain.mentions:
            key = (mention.sentence, mention.start_word, mention.end_word)
            if key == rep_key:
                continue
            mention_to_rep[key] = rep_text

    # Apply replacements; for multi-word mentions we replace the span with a single token
    resolved_tokens = []
    idx = 0
    while idx < len(token_lookup):
        sent_idx, word_idx = token_lookup[idx]

        # Find any mention starting at this position
        replacement = None
        replacement_span_len = 1
        for (m_sent, m_start, m_end), rep_text in mention_to_rep.items():
            if m_sent == sent_idx and m_start == word_idx:
                replacement = rep_text
                replacement_span_len = max(1, m_end - m_start)
                break

        if replacement is not None:
            resolved_tokens.append(replacement)
            idx += replacement_span_len
        else:
            resolved_tokens.append(tokens[idx])
            idx += 1

    return resolved_tokens

