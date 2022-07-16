"""
Utility functions for dealing with NER tagging.
"""

import logging

logger = logging.getLogger('stanza')

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
        if tag == 'O':
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
        if tag == 'O':
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
        if tag == 'O':
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
        if tag == 'O':
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

def process_tags(sentences, scheme):
    res = []
    # check if tag conversion is needed
    convert_bio_to_bioes = False
    convert_basic_to_bioes = False
    is_bio = is_bio_scheme([x[1] for sent in sentences for x in sent])
    is_basic = not is_bio and is_basic_scheme([x[1] for sent in sentences for x in sent])
    if is_bio and scheme.lower() == 'bioes':
        convert_bio_to_bioes = True
        logger.debug("BIO tagging scheme found in input; converting into BIOES scheme...")
    elif is_basic and scheme.lower() == 'bioes':
        convert_basic_to_bioes = True
        logger.debug("Basic tagging scheme found in input; converting into BIOES scheme...")
    # process tags
    for sent in sentences:
        words, tags = zip(*sent)
        # NER field sanity checking
        if any([x is None or x == '_' for x in tags]):
            raise ValueError("NER tag not found for some input data.")
        if convert_basic_to_bioes:
            # if basic, convert tags -> bio -> bioes
            tags = bio2_to_bioes(basic_to_bio(tags))
        else:
            # first ensure BIO2 scheme
            tags = to_bio2(tags)
            # then convert to BIOES
            if convert_bio_to_bioes:
                tags = bio2_to_bioes(tags)
        res.append([(w,t) for w,t in zip(words, tags)])
    return res


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
