"""
Utility functions for NER.
"""

def is_bio_scheme(all_tags):
    """
    Check if BIO tagging scheme is used. Return True if so.
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
