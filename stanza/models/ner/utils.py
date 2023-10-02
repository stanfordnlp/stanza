"""
Utility functions for dealing with NER tagging.
"""

import logging

from stanza.models.common.vocab import EMPTY

logger = logging.getLogger('stanza')

EMPTY_TAG = ('_', '-', '', None)
EMPTY_OR_O_TAG = tuple(list(EMPTY_TAG) + ['O'])

def is_basic_scheme(all_tags):
    """
    Check if a basic tagging scheme is used. Return True if so.

    Args:
        all_tags: a list of NER tags

    Returns:
        True if the tagging scheme does not use B-, I-, etc, otherwise False
    """
    for tag in all_tags:
        if len(tag) > 2 and tag[:2] in ('B-', 'I-', 'S-', 'E-', 'B_', 'I_', 'S_', 'E_'):
            return False
    return True


def is_bio_scheme(all_tags):
    """
    Check if BIO tagging scheme is used. Return True if so.

    Args:
        all_tags: a list of NER tags
    
    Returns:
        True if the tagging scheme is BIO, otherwise False
    """
    for tag in all_tags:
        if tag in EMPTY_OR_O_TAG:
            continue
        elif len(tag) > 2 and tag[:2] in ('B-', 'I-', 'B_', 'I_'):
            continue
        else:
            return False
    return True

def to_bio2(tags):
    """
    Convert the original tag sequence to BIO2 format. If the input is already in BIO2 format,
    the original input is returned.

    Args:
        tags: a list of tags in either BIO or BIO2 format
    
    Returns:
        new_tags: a list of tags in BIO2 format
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag in EMPTY_OR_O_TAG:
            new_tags.append(tag)
        elif tag[0] == 'I':
            if i == 0 or tags[i-1] == 'O' or tags[i-1][1:] != tag[1:]:
                new_tags.append('B' + tag[1:])
            else:
                new_tags.append(tag)
        else:
            new_tags.append(tag)
    return new_tags

def basic_to_bio(tags):
    """
    Convert a basic tag sequence into a BIO sequence.
    You can compose this with bio2_to_bioes to convert to bioes

    Args:
        tags: a list of tags in basic (no B-, I-, etc) format

    Returns:
        new_tags: a list of tags in BIO format
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag in EMPTY_OR_O_TAG:
            new_tags.append(tag)
        elif i == 0 or tags[i-1] == 'O' or tags[i-1] != tag:
            new_tags.append('B-' + tag)
        else:
            new_tags.append('I-' + tag)
    return new_tags


def bio2_to_bioes(tags):
    """
    Convert the BIO2 tag sequence into a BIOES sequence.

    Args:
        tags: a list of tags in BIO2 format

    Returns:
        new_tags: a list of tags in BIOES format
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag in EMPTY_OR_O_TAG:
            new_tags.append(tag)
        else:
            if len(tag) < 2:
                raise Exception(f"Invalid BIO2 tag found: {tag}")
            else:
                if tag[:2] in ('I-', 'I_'): # convert to E- if next tag is not I-
                    if i+1 < len(tags) and tags[i+1][:2] in ('I-', 'I_'):
                        new_tags.append('I-' + tag[2:]) # compensate for underscores
                    else:
                        new_tags.append('E-' + tag[2:])
                elif tag[:2] in ('B-', 'B_'): # convert to S- if next tag is not I-
                    if i+1 < len(tags) and tags[i+1][:2] in ('I-', 'I_'):
                        new_tags.append('B-' + tag[2:])
                    else:
                        new_tags.append('S-' + tag[2:])
                else:
                    raise Exception(f"Invalid IOB tag found: {tag}")
    return new_tags

def normalize_empty_tags(sentences):
    """
    If any tags are None, _, -, or blank, turn them into EMPTY

    The input should be a list(sentence) of list(word) of tuple(text, list(tag))
    which is the typical format for the data at the time data.py is preprocessing the tags
    """
    new_sentences = [[(word[0], tuple(EMPTY if x in EMPTY_TAG else x for x in word[1])) for word in sentence]
                     for sentence in sentences]
    return new_sentences

def process_tags(sentences, scheme):
    """
    Convert tags in these sentences to bioes

    We allow empty tags ('_', '-', None), which will represent tags
    that do not get any gradient when training
    """
    all_words = []
    all_tags = []
    converted_tuples = False
    for sent_idx, sent in enumerate(sentences):
        words, tags = zip(*sent)
        all_words.append(words)
        # if we got one dimension tags w/o tuples or lists, make them tuples
        # but we also check that the format is consistent,
        # as otherwise the result being converted might be confusing
        if not converted_tuples and any(tag is None or isinstance(tag, str) for tag in tags):
            if sent_idx > 0:
                raise ValueError("Got a mix of tags and lists of tags.  First non-list was in sentence %d" % sent_idx)
            converted_tuples = True
        if converted_tuples:
            if not all(tag is None or isinstance(tag, str) for tag in tags):
                raise ValueError("Got a mix of tags and lists of tags.  First tag as a list was in sentence %d" % sent_idx)
            tags = [(tag,) for tag in tags]
        all_tags.append(tags)

    max_columns = max(len(x) for tags in all_tags for x in tags)
    for sent_idx, tags in enumerate(all_tags):
        if any(len(x) < max_columns for x in tags):
            raise ValueError("NER tags not uniform in length at sentence %d.  TODO: extend those columns with O" % sent_idx)

    all_convert_bio_to_bioes = []
    all_convert_basic_to_bioes = []

    for column_idx in range(max_columns):
        # check if tag conversion is needed for each column
        # we treat each column separately, although practically
        # speaking it would be pretty weird for a dataset to have BIO
        # in one column and basic in another, for example
        convert_bio_to_bioes = False
        convert_basic_to_bioes = False
        tag_column = [x[column_idx] for sent in all_tags for x in sent]
        is_bio = is_bio_scheme(tag_column)
        is_basic = not is_bio and is_basic_scheme(tag_column)
        if is_bio and scheme.lower() == 'bioes':
            convert_bio_to_bioes = True
            logger.debug("BIO tagging scheme found in input at column %d; converting into BIOES scheme..." % column_idx)
        elif is_basic and scheme.lower() == 'bioes':
            convert_basic_to_bioes = True
            logger.debug("Basic tagging scheme found in input at column %d; converting into BIOES scheme..." % column_idx)
        all_convert_bio_to_bioes.append(convert_bio_to_bioes)
        all_convert_basic_to_bioes.append(convert_basic_to_bioes)

    result = []
    for words, tags in zip(all_words, all_tags):
        # process tags
        # tags is a list of each column of tags for each word in this sentence
        # copy the tags to a list so we can edit them
        tags = [[x for x in sentence_tags] for sentence_tags in tags]
        for column_idx, (convert_bio_to_bioes, convert_basic_to_bioes) in enumerate(zip(all_convert_bio_to_bioes, all_convert_basic_to_bioes)):
            tag_column = [x[column_idx] for x in tags]
            if convert_basic_to_bioes:
                # if basic, convert tags -> bio -> bioes
                tag_column = bio2_to_bioes(basic_to_bio(tag_column))
            else:
                # first ensure BIO2 scheme
                tag_column = to_bio2(tag_column)
                # then convert to BIOES
                if convert_bio_to_bioes:
                    tag_column = bio2_to_bioes(tag_column)
            for tag_idx, tag in enumerate(tag_column):
                tags[tag_idx][column_idx] = tag
        result.append([(w,tuple(t)) for w,t in zip(words, tags)])

    if converted_tuples:
        result = [[(word[0], word[1][0]) for word in sentence] for sentence in result]
    return result


def decode_from_bioes(tags):
    """
    Decode from a sequence of BIOES tags, assuming default tag is 'O'.
    Args:
        tags: a list of BIOES tags
    
    Returns:
        A list of dict with start_idx, end_idx, and type values.
    """
    res = []
    ent_idxs = []
    cur_type = None

    def flush():
        if len(ent_idxs) > 0:
            res.append({
                'start': ent_idxs[0], 
                'end': ent_idxs[-1], 
                'type': cur_type})

    for idx, tag in enumerate(tags):
        if tag is None:
            tag = 'O'
        if tag == 'O':
            flush()
            ent_idxs = []
        elif tag.startswith('B-'): # start of new ent
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
        elif tag.startswith('I-'): # continue last ent
            ent_idxs.append(idx)
            cur_type = tag[2:]
        elif tag.startswith('E-'): # end last ent
            ent_idxs.append(idx)
            cur_type = tag[2:]
            flush()
            ent_idxs = []
        elif tag.startswith('S-'): # start single word ent
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
            flush()
            ent_idxs = []
    # flush after whole sentence
    flush()
    return res


def merge_tags(*sequences):
    """
    Merge multiple sequences of NER tags into one sequence

    Only O is replaced, and the earlier tags have precedence
    """
    tags = list(sequences[0])
    for sequence in sequences[1:]:
        idx = 0
        while idx < len(sequence):
            # skip empty tags in the later sequences
            if sequence[idx] == 'O':
                idx += 1
                continue

            # check for singletons.  copy if not O in the original
            if sequence[idx].startswith("S-"):
                if tags[idx] == 'O':
                    tags[idx] = sequence[idx]
                idx += 1
                continue

            # at this point, we know we have a B-... sequence
            if not sequence[idx].startswith("B-"):
                raise ValueError("Got unexpected tag sequence at idx {}: {}".format(idx, sequence))

            # take the block of tags which are B- through E-
            start_idx = idx
            end_idx = start_idx + 1
            while end_idx < len(sequence):
                if sequence[end_idx][2:] != sequence[start_idx][2:]:
                    raise ValueError("Unexpected tag sequence at idx {}: {}".format(end_idx, sequence))
                if sequence[end_idx].startswith("E-"):
                    break
                if not sequence[end_idx].startswith("I-"):
                    raise ValueError("Unexpected tag sequence at idx {}: {}".format(end_idx, sequence))
                end_idx += 1
            if end_idx == len(sequence):
                raise ValueError("Got a sequence with an unclosed tag: {}".format(sequence))
            end_idx = end_idx + 1

            # if all tags in the original are O, we can overwrite
            # otherwise, keep the originals
            if all(x == 'O' for x in tags[start_idx:end_idx]):
                tags[start_idx:end_idx] = sequence[start_idx:end_idx]
            idx = end_idx

    return tags
