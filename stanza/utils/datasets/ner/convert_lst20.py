"""
Converts the Thai LST20 dataset to a format usable by Stanza's NER model

The dataset in the original format has a few tag errors which we
automatically fix (or at worst cover up)
"""

import os

from stanza.utils.datasets.ner.utils import convert_bio_to_json

def convert_lst20(paths, short_name, include_space_char=True):
    assert short_name == "th_lst20"
    SHARDS = ("train", "eval", "test")
    BASE_OUTPUT_PATH = paths["NER_DATA_DIR"]

    input_split = [(os.path.join(paths["NERBASE"], "thai", "LST20_Corpus", x), x) for x in SHARDS]

    if not include_space_char:
        short_name = short_name + "_no_ws"

    for input_folder, split_type in input_split:
        text_list = [text for text in os.listdir(input_folder) if text[0] == 'T']

        if split_type == "eval":
            split_type = "dev"

        output_path = os.path.join(BASE_OUTPUT_PATH, "%s.%s.bio" % (short_name, split_type))
        print(output_path)

        with open(output_path, 'w', encoding='utf-8') as fout:
            for text in text_list:
                lst = []
                with open(os.path.join(input_folder, text), 'r', encoding='utf-8') as fin:
                    lines = fin.readlines()

                for line_idx, line in enumerate(lines):
                    x = line.strip().split('\t')
                    if len(x) > 1:
                        if x[0] == '_' and not include_space_char:
                            continue
                        else:
                            word, tag = x[0], x[2]

                            if tag == "MEA_BI":
                                tag = "B_MEA"
                            if tag == "OBRN_B":
                                tag = "B_BRN"
                            if tag == "ORG_I":
                                tag = "I_ORG"
                            if tag == "PER_I":
                                tag = "I_PER"
                            if tag == "LOC_I":
                                tag = "I_LOC"
                            if tag == "B" and line_idx + 1 < len(lines):
                                x_next = lines[line_idx+1].strip().split('\t')
                                if len(x_next) > 1:
                                    tag_next = x_next[2]
                                    if "I_" in tag_next or "E_" in tag_next:
                                        tag = tag + tag_next[1:]
                                    else:
                                        tag = "O"
                                else:
                                    tag = "O"
                            if "_" in tag:
                                tag = tag.replace("_", "-")
                            if "ABB" in tag or tag == "DDEM" or tag == "I" or tag == "__":
                                tag = "O"

                            fout.write('{}\t{}'.format(word, tag))
                            fout.write('\n')
                    else:
                        fout.write('\n')
    convert_bio_to_json(BASE_OUTPUT_PATH, BASE_OUTPUT_PATH, short_name)
