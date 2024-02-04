from collections import defaultdict
from functools import lru_cache
import json
import os
import re

import stanza

from stanza.models.constituency import tree_reader
from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm

from stanza.utils.conll import CoNLL

tqdm = get_tqdm()

# TODO: move this to a utility module and try it on other languages
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

def process_documents(docs):
    processed_section = []

    for idx, (doc, doc_id) in enumerate(tqdm(docs)):
        # extract the entities
        # get sentence words and lengths
        sentences = [[j.text for j in i.words]
                    for i in doc.sentences]
        sentence_lens = [len(x.words) for x in doc.sentences]
        cased_words = [y for x in sentences for y in x]
        sent_id = [y for idx, sent_len in enumerate(sentence_lens) for y in [idx] * sent_len]

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
        SPANS = re.compile("(\(\w+|[%\w]+\))")
        for parsed_sentence in doc.sentences:
            # spans regex
            # parse the misc column, leaving on "Entity" entries
            misc = [[k.split("=")
                    for k in j
                    if k.split("=")[0] == "Entity"]
                    for i in parsed_sentence.words
                    for j in [i.misc.split("|") if i.misc else []]]
            # and extract the Entity entry values
            entities = [i[0][1] if len(i) > 0 else None for i in misc]
            # extract reference information
            refs = [SPANS.findall(i) if i else [] for i in entities]
            # and calculate spans: the basic rule is (e... begins a reference
            # and ) without e before ends the most recent reference
            # every single time we get a closing element, we pop it off
            # the refdict and insert the pair to final_refs
            refdict = defaultdict(list)
            final_refs = defaultdict(list)
            last_ref = None
            for indx, i in enumerate(refs):
                for j in i:
                    # this is the beginning of a reference
                    if j[0] == "(":
                        refdict[j[1:]].append(indx)
                        last_ref = j[1:]
                    # at the end of a reference, if we got exxxxx, that ends
                    # a particular refereenc; otherwise, it ends the last reference
                    elif j[-1] == ")" and j[:-1].isnumeric():
                        final_refs[j[:-1]].append((refdict[j[:-1]].pop(-1), indx))
                    elif j[-1] == ")":
                        final_refs[last_ref].append((refdict[last_ref].pop(-1), indx))
                        last_ref = None
            final_refs = dict(final_refs)
            # convert it to the right format (specifically, in (ref, start, end) tuples)
            coref_spans = []
            for k, v in final_refs.items():
                for i in v:
                    coref_spans.append([int(k), i[0], i[1]])
            sentence_upos = [x.upos for x in parsed_sentence.words]
            sentence_heads = [x.head - 1 if x.head > 0 else None for x in parsed_sentence.words]
            for span in coref_spans:
                # input is expected to be start word, end word + 1
                # counting from 0
                # whereas the OntoNotes coref_span is [start_word, end_word] inclusive
                span_start = span[1] + word_total
                span_end = span[2] + word_total + 1
                candidate_head = find_cconj_head(sentence_heads, sentence_upos, span[1], span[2]+1)
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
            word_total += len(parsed_sentence.words)
        span_clusters = sorted([sorted(values) for _, values in span_clusters.items()])
        word_clusters = sorted([sorted(values) for _, values in word_clusters.items()])
        head2span = sorted(head2span)

        processed = {
            "document_id": doc_id,
            "cased_words": cased_words,
            "sent_id": sent_id,
            "part_id": idx,
            # "pos": pos,
            "deprel": deprel,
            "head": heads,
            "span_clusters": span_clusters,
            "word_clusters": word_clusters,
            "head2span": head2span,
        }
        processed_section.append(processed)
    return processed_section

SECTION_NAMES = ["train", "dev", "test"]
SHORT_NAME = "en_gum-ud"
LANGUAGE = "english"

def process_dataset(short_name, conllu_path, coref_output_path):

    for section in SECTION_NAMES:
        load = os.path.join(conllu_path, f"{short_name}-{section}.conllu")
        print("Processing %s from %s" % (section, load))
        input_file = CoNLL.conll2multi_docs(load, return_doc_ids=True)
        converted_section = process_documents(input_file)

        output_filename = os.path.join(coref_output_path, "%s.%s.json" % (short_name, section))
        with open(output_filename, "w", encoding="utf-8") as fout:
            json.dump(converted_section, fout, indent=2)

def main():
    paths = get_default_paths()
    coref_input_path = paths['COREF_BASE']
    conll_path = os.path.join(coref_input_path, LANGUAGE, SHORT_NAME)
    coref_output_path = paths['COREF_DATA_DIR']
    process_dataset(SHORT_NAME, conll_path, coref_output_path)

if __name__ == '__main__':
    main()

# docs = CoNLL.conll2multi_docs("./en_gum-corefud-dev.conllu", return_doc_ids=True)
# docs[0][1]

# head2span
# word_clusters


# sent = doc.sentences[1]

# {
#     "document_id": doc_id,
#     "cased_words": cased_words,
#     "sent_id": sent_id,
#     "part_id": part_id,
#     "speaker": speaker,
#     #"pos": pos,
#             "deprel": deprel,
#     "head": heads,
#     "span_clusters": span_clusters,
#     "word_clusters": word_clusters,
#     "head2span": head2span,
# }
