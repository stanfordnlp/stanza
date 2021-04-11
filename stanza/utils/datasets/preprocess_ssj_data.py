"""
The SSJ dataset has an unusual bug: all of the sentences end with SpaceAfter=no

This script fixes them and writes the fixed files to the given location.
"""

import stanza.utils.datasets.common as common

def process(input_conllu, input_txt_copy, input_conllu_copy):
    conllu_lines = open(input_conllu).readlines()

    new_conllu_lines = list(conllu_lines)

    line_idx = 0
    text_idx = 0
    # invariant: conllu_lines[line_idx] is
    #  # sent_id
    # at the start of a loop
    while line_idx < len(conllu_lines):
        # extract the text from the comments before each sentence
        line_idx = line_idx + 1
        text_line = conllu_lines[line_idx]
        assert text_line.startswith("# text = "), "Unexpected format: %s,%d is not # text" % (input_txt, line_idx)
        text_line = text_line[9:-1]
        # use that text to keep track of an index in the text where we might need to put new spaces
        text_idx = text_idx + len(text_line)

        # advance to the end of the sentence
        line_idx = line_idx + 1
        assert conllu_lines[line_idx].startswith("1"), "Unexpected format: %s,%d is not a word" % (input_txt, line_idx)
        while conllu_lines[line_idx].strip():
            line_idx = line_idx + 1
        last_word_idx = line_idx - 1
        
        # check if the end of the sentence has SpaceAfter or not
        new_line = conllu_lines[last_word_idx].replace("SpaceAfter=No|", "")
        assert new_line.find("SpaceAfter=") < 0, "Unexpected format: %s,%d has unusual SpaceAfter" % (input_txt, line_idx)

        # if not, need to add a new space
        if new_line != conllu_lines[last_word_idx]:
            conllu_lines[last_word_idx] = new_line
        text_idx = text_idx + 1
        
        # done with a sentence.  skip to the start of the next sentence
        # or the end of the document
        while line_idx < len(conllu_lines) and not conllu_lines[line_idx].strip():
            line_idx = line_idx + 1

    with open(input_conllu_copy, "w") as fout:
        for line in conllu_lines:
            fout.write(line)

    common.convert_conllu_to_txt(input_conllu_copy, input_txt_copy)
