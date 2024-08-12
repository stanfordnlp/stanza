from collections import defaultdict
from functools import lru_cache

class DynamicDepth():
    """
    Implements a cache + dynamic programming to find the relative depth of every word in a subphrase given the head word for every word.
    """
    def get_parse_depths(self, heads, start, end):
        """Return the relative depth for every word

        Args:
            heads (list): List where each entry is the index of that entry's head word in the dependency parse
            start (int): starting index of the heads for the subphrase
            end (int): ending index of the heads for the subphrase

        Returns:
            list: Relative depth in the dependency parse for every word
        """
        self.heads = heads[start:end]
        self.relative_heads = [h - start if h else -100 for h in self.heads] # -100 to deal with 'none' headwords

        depths = [self._get_depth_recursive(h) for h in range(len(self.relative_heads))]

        return depths

    @lru_cache(maxsize=None)
    def _get_depth_recursive(self, index):
        """Recursively get the depths of every index using a cache and recursion

        Args:
            index (int): Index of the word for which to calculate the relative depth

        Returns:
            int: Relative depth of the word at the index
        """
        # if the head for the current index is outside the scope, this index is a relative root
        if self.relative_heads[index] >= len(self.relative_heads) or self.relative_heads[index] < 0:
            return 0
        return self._get_depth_recursive(self.relative_heads[index]) + 1

def find_cconj_head(heads, upos, start, end):
    """
    Finds how far each word is from the head of a span, then uses the closest CCONJ to the head as the new head

    If no CCONJ is present, returns None
    """
    # use head information to extract parse depth
    dynamicDepth = DynamicDepth()
    depth = dynamicDepth.get_parse_depths(heads, start, end)
    depth_limit = 2

    # return first 'CCONJ' token above depth limit, if exists
    # unlike the original paper, we expect the parses to use UPOS, hence CCONJ instead of CC
    cc_indexes = [i for i in range(end - start) if upos[i+start] == 'CCONJ' and depth[i] < depth_limit]
    if cc_indexes:
        return cc_indexes[0] + start
    return None

def process_document(pipe, doc_id, part_id, sentences, coref_spans, sentence_speakers, use_cconj_heads=True):
    """
    coref_spans: a list of lists
    one list per sentence
    each sentence has a list of spans, where each span is (span_index, span_start, span_end)
    """
    sentence_lens = [len(x) for x in sentences]
    if all(isinstance(x, list) for x in sentence_speakers):
        speaker = [y for x in sentence_speakers for y in x]
    else:
        speaker = [y for x, sent_len in zip(sentence_speakers, sentence_lens) for y in [x] * sent_len]

    cased_words = [y for x in sentences for y in x]
    sent_id = [y for idx, sent_len in enumerate(sentence_lens) for y in [idx] * sent_len]

    # use the trees to get the xpos tags
    # alternatively, could translate the pos_tags field,
    # but those have numbers, which is annoying
    #tree_text = "\n".join(x['parse_tree'] for x in paragraph)
    #trees = tree_reader.read_trees(tree_text)
    #pos = [x.label for tree in trees for x in tree.yield_preterminals()]
    # actually, the downstream code doesn't use pos at all.  maybe we can skip?

    doc = pipe(sentences)
    word_total = 0
    heads = []
    # TODO: does SD vs UD matter?
    deprel = []
    for sentence in doc.sentences:
        for word in sentence.words:
            deprel.append(word.deprel)
            if word.head == 0:
                heads.append("null")
            else:
                heads.append(word.head - 1 + word_total)
        word_total += len(sentence.words)

    span_clusters = defaultdict(list)
    word_clusters = defaultdict(list)
    head2span = []
    word_total = 0
    for parsed_sentence, ontonotes_coref, ontonotes_words in zip(doc.sentences, coref_spans, sentences):
        sentence_upos = [x.upos for x in parsed_sentence.words]
        sentence_heads = [x.head - 1 if x.head > 0 else None for x in parsed_sentence.words]
        for span in ontonotes_coref:
            # input is expected to be start word, end word + 1
            # counting from 0
            # whereas the OntoNotes coref_span is [start_word, end_word] inclusive
            span_start = span[1] + word_total
            span_end = span[2] + word_total + 1
            candidate_head = find_cconj_head(sentence_heads, sentence_upos, span[1], span[2]+1) if use_cconj_heads else None
            if candidate_head is None:
                for candidate_head in range(span[1], span[2] + 1):
                    # stanza uses 0 to mark the head, whereas OntoNotes is counting
                    # words from 0, so we have to subtract 1 from the stanza heads
                    #print(span, candidate_head, parsed_sentence.words[candidate_head].head - 1)
                    # treat the head of the phrase as the first word that has a head outside the phrase
                    if (parsed_sentence.words[candidate_head].head - 1 < span[1] or
                        parsed_sentence.words[candidate_head].head - 1 > span[2]):
                        break
                else:
                    # if none have a head outside the phrase (circular??)
                    # then just take the first word
                    candidate_head = span[1]
            #print("----> %d" % candidate_head)
            candidate_head += word_total
            span_clusters[span[0]].append((span_start, span_end))
            word_clusters[span[0]].append(candidate_head)
            head2span.append((candidate_head, span_start, span_end))
        word_total += len(ontonotes_words)
    span_clusters = sorted([sorted(values) for _, values in span_clusters.items()])
    word_clusters = sorted([sorted(values) for _, values in word_clusters.items()])
    head2span = sorted(head2span)

    processed = {
        "document_id": doc_id,
        "part_id": part_id,
        "cased_words": cased_words,
        "sent_id": sent_id,
        "speaker": speaker,
        #"pos": pos,
        "deprel": deprel,
        "head": heads,
        "span_clusters": span_clusters,
        "word_clusters": word_clusters,
        "head2span": head2span,
    }
    if part_id is not None:
        processed["part_id"] = part_id
    return processed
