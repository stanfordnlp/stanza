"""
Utility functions for NER.
"""

def is_bio_scheme(all_tags):
    """
    Check all tags to see if BIO tagging scheme is used. Return True if so.
    """
    all_prefix = set([t[0].lower() for t in all_tags])
    if all_prefix == set(['b', 'i', 'o']):
        return True
    else:
        return False

def convert_tags_to_bioes(tags):
    """
    Convert the original BIO tag sequence into an BIOES scheme.
    """
    new_tags = []
    pre, cur, nex = '', '', ''
    new_t = ''
    # handle single token sentence
    if len(tags) == 1:
        if tags[0].startswith('I'):
            new_tags.append(tags[0].replace('I-', 'S-'))
        else:
            new_tags.append(tags[0])
        return new_tags
    # sentences that have >=2 tokens
    for i, t in enumerate(tags):
        pre = cur if i > 0 else ''
        cur = t
        nex = tags[i+1] if i < len(tags)-1 else ''
        if i == 0: # first tag
            if cur.startswith('I'):
                if is_different_chunk(cur, nex):
                    new_t = cur.replace('I-', 'S-')
                else:
                    new_t = cur.replace('I-', 'B-')
            else: # cur == 'O'
                new_t = cur
        elif i == len(tags) - 1: # last tag
            if cur.startswith('I'):
                if is_different_chunk(pre,cur):
                    new_t = cur.replace('I-', 'S-')
                else:
                    new_t = cur.replace('I-', 'E-')
            elif cur.startswith('B'):
                new_t = cur.replace('B-', 'S-')
            else: # cur == 'O'
                new_t = cur
        else:
            if cur.startswith('I'):
                if is_different_chunk(pre, cur):
                    if is_different_chunk(cur, nex):
                        new_t = cur.replace('I-', 'S-')
                    else:
                        new_t = cur.replace('I-', 'B-')
                else:
                    if is_different_chunk(cur, nex):
                        new_t = cur.replace('I-', 'E-')
                    else:
                        new_t = cur
            elif cur.startswith('B'):
                if is_different_chunk(cur, nex):
                    new_t = cur.replace('B-', 'S-')
                else:
                    new_t = cur
            else: # cur == 'O'
                new_t = cur
        new_tags.append(new_t)
    return new_tags

def is_different_chunk(tag1, tag2):
    """ tag1 must come before tag2 in sequence. """
    if tag1 == 'O' and tag2 == 'O':
        return False
    if tag1 == 'O' and tag2 != 'O':
        return True
    if tag2 == 'O' and tag1 != 'O':
        return True
    if tag1.startswith('I') and tag2.startswith('B'):
        return True
    if tag1.startswith('I') and tag2.startswith('I'):
        return (tag1[2:] != tag2[2:])
    return False


