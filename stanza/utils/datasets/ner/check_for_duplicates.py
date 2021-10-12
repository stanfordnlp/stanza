"""
A simple tool to check if there are duplicates in a set of NER files

It's surprising how many datasets have a bunch of duplicates...
"""

def read_sentences(filename):
    """
    Read the sentences (without tags) from a BIO file
    """
    sentences = []
    with open(filename) as fin:
        lines = fin.readlines()
    current_sentence = []
    for line in lines:
        line = line.strip()
        if not line:
            if current_sentence:
                sentences.append(tuple(current_sentence))
            current_sentence = []
            continue
        word = line.split("\t")[0]
        current_sentence.append(word)
    if len(current_sentence) > 0:
        sentences.append(tuple(current_sentence))
    return sentences
    
def check_for_duplicates(output_filenames, fail=False, check_self=False, print_all=False):
    """
    Checks for exact duplicates in a list of NER files
    """
    sentence_map = {}
    for output_filename in output_filenames:
        duplicates = 0
        sentences = read_sentences(output_filename)
        for sentence in sentences:
            other_file = sentence_map.get(sentence, None)
            if other_file is not None and (check_self or other_file != output_filename):
                if fail:
                    raise ValueError("Duplicate sentence '{}', first in {}, also in {}".format("".join(sentence), sentence_map[sentence], output_filename))
                else:
                    if duplicates == 0 and not print_all:
                        print("First duplicate:")
                    if duplicates == 0 or print_all:                    
                        print("{}\nFound in {} and {}".format(sentence, other_file, output_filename))
                    duplicates = duplicates + 1
            sentence_map[sentence] = output_filename
        if duplicates > 0:
            print("%d duplicates found in %s" % (duplicates, output_filename))
