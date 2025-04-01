"""
merge_multiple_datasets.py
--------------------------
This script merges multiple datasets into one. It is assumed that the datasets are
stored in the following directory structure:
        - combined/
                - split1/
                - singletons/
                        - file1.json
                        - file2.json
                        ...
                - no_singletons/
                        - file1.json
                        - file2.json
                        ...
                - split2/
                - singletons/
                        - file1.json
                        - file2.json
                        ...
                - no_singletons/
                        - file1.json
                        - file2.json
                        ...
                ...

The script will merge all the files in the singletons and no_singletons directories of
each split into a single file and save it in the root of the combined directory. The
merged files will be named as combined.split1.json, combined.split2.json, etc.

Importantly, the script will annotate for whether the data files contains signletons
or not, and annotate them appropriatly in the merged file.
"""

from pathlib import Path
from glob import glob
import json

DATA_PATH = Path("./combined/")
SPLITS = [Path(i).stem for i in glob(str(DATA_PATH/"*"))]

for i in SPLITS:
    singletons = glob(str(DATA_PATH/i/"singletons/*"))
    no_singletons = glob(str(DATA_PATH/i/"no_singletons/*"))

    all = []
    for j in singletons:
        with open(j, 'r') as df:
            data = json.load(df)
        for k in data:
            k["singletons"] = True
        all += data

    for j in no_singletons:
        with open(j, 'r') as df:
            data = json.load(df)
        for k in data:
            k["singletons"] = False
        all += data

    with open(DATA_PATH.stem+"."+i+".json", 'w') as df:
        json.dump(all, df, indent=2)
