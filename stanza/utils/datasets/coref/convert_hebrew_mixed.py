"""
Build a dataset mixed with IAHLT Hebrew and UD Coref

We find that the IAHLT dataset by itself, trained using Stanza 1.11
with xlm-roberta-large and a lora finetuning layer, gets 49.7 F1.
This is a bit lower than the value the IAHLT group originally had, as
they reported 52.  Interestingly, we find that mixing in the 1.3 UD
Coref improves results, getting 51.7 under the same parameters

This script runs the IAHLT conversion and the UD Coref conversion,
then combines the files into one big training file
"""

import json
import os
import shutil
import tempfile

from stanza.utils.datasets.coref import convert_hebrew_iahlt
from stanza.utils.datasets.coref import convert_udcoref
from stanza.utils.default_paths import get_default_paths

def main():
    paths = get_default_paths()
    coref_output_path = paths['COREF_DATA_DIR']
    with tempfile.TemporaryDirectory() as temp_dir_path:
        hebrew_filenames = convert_hebrew_iahlt.main(["--output_directory", temp_dir_path])
        udcoref_filenames = convert_udcoref.main(["--project", "gerrom", "--output_directory", temp_dir_path])

        with open(os.path.join(temp_dir_path, hebrew_filenames[0]), encoding="utf-8") as fin:
            hebrew_train = json.load(fin)
        udcoref_train_filename = os.path.join(temp_dir_path, udcoref_filenames[0])
        with open(udcoref_train_filename, encoding="utf-8") as fin:
            print("Reading extra udcoref json data from %s" % udcoref_train_filename)
            udcoref_train = json.load(fin)
        mixed_train = hebrew_train + udcoref_train
        with open(os.path.join(coref_output_path, "he_mixed.train.json"), "w", encoding="utf-8") as fout:
            json.dump(mixed_train, fout, indent=2, ensure_ascii=False)

        shutil.copyfile(os.path.join(temp_dir_path, hebrew_filenames[1]),
                        os.path.join(coref_output_path, "he_mixed.dev.json"))
        shutil.copyfile(os.path.join(temp_dir_path, hebrew_filenames[2]),
                        os.path.join(coref_output_path, "he_mixed.test.json"))

if __name__ == '__main__':
    main()
