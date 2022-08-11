"""
Converts the Thai NNER22 dataset to a nested format for Stanza's NER model

The dataset is already written in json format, so we will convert into a compatible json format.

The dataset in the original format has nested NER format using span key to indicate the oken span. We will extract all the layers
of NER tag and write it as a list of tags which will be fed into Stanza model
"""

import os
import logging
import json

def convert_nner22(paths, short_name, include_space_char=True):
    assert short_name == "th_nner22"
    SHARDS = ("train", "dev", "test")
    BASE_INPUT_PATH = os.path.join(paths["NERBASE"], "thai", "Thai-NNER", "data", "scb-nner-th-2022", "postproc")

    if not include_space_char:
        short_name = short_name + "_no_ws"

    for shard in SHARDS:
        input_path = os.path.join(BASE_INPUT_PATH, "%s.json" % (shard))
        output_path = os.path.join(paths["NER_DATA_DIR"], "ner", "%s.%s.json" % (short_name, shard))

        logging.info("Output path for %s split at %s" % (shard, output_path))

        data = json.load(open(input_path))

        documents = []

        for i in range(len(data)):
            token, entities = data[i]["tokens"], data[i]["entities"]

            token_length = len(token)
            document, ner_dict = [], {}

            for entity in entities:
                start, stop = entity["span"]
                ner = entity["entity_type"].upper()
                
                for j in range(start, stop):
                    if j == start:
                        ner_tag = "B-" + ner
                    elif j == stop - 1:
                        ner_tag = "E-" + ner
                    else:
                        ner_tag = "I-" + ner

                    if j not in ner_dict:
                        ner_dict[j] = []
                    ner_dict[j].append(ner_tag)

            for k in range(token_length):
                dict_add = {}

                if k not in ner_dict:
                    dict_add["ner"] = ["O"]
                else:
                    dict_add["ner"] = ner_dict[k]
                dict_add["text"] = token[k]
                
                document.append(dict_add)

            documents.append(document)

        with open(output_path, "w") as outfile:
            json.dump(documents, outfile, indent=1)

        logging.info("%s.%s.json file successfully created" % (short_name, shard))
