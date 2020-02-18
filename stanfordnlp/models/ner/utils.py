"""
Utility functions for dealing with NER tagging.
"""

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
        elif len(tag) > 2 and tag[:2] in ('B-', 'I-'):
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
                if tag[:2] == 'I-': # convert to E- if next tag is not I-
                    if i+1 < len(tags) and tags[i+1][:2] == 'I-':
                        new_tags.append(tag)
                    else:
                        new_tags.append('E-' + tag[2:])
                elif tag[:2] == 'B-': # convert to S- if next tag is not I-
                    if i+1 < len(tags) and tags[i+1][:2] == 'I-':
                        new_tags.append(tag)
                    else:
                        new_tags.append('S-' + tag[2:])
                else:
                    raise Exception(f"Invalid IOB tag found: {tag}")
    return new_tags

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
