import glob
import os
import re

import xml.etree.ElementTree as ET

from stanza.models.constituency import tree_reader
from stanza.utils.datasets.constituency.utils import write_dataset

def filenum_to_shard(filenum):
    if filenum >= 1 and filenum <= 815:
        return 0
    if filenum >= 1001 and filenum <= 1136:
        return 0

    if filenum >= 886 and filenum <= 931:
        return 1
    if filenum >= 1148 and filenum <= 1151:
        return 1

    if filenum >= 816 and filenum <= 885:
        return 2
    if filenum >= 1137 and filenum <= 1147:
        return 2

    raise ValueError("Unhandled filenum %d" % filenum)

def collect_trees(root):
    if root.tag == 'S':
        yield root.text, root.attrib['ID']

    for child in root:
        for tree in collect_trees(child):
            yield tree

id_re = re.compile("<S ID=([0-9a-z]+)>")
amp_re = re.compile("[&]")

def convert_ctb(input_dir, output_dir, dataset_name):
    input_files = glob.glob(os.path.join(input_dir, "*"))

    # train, dev, test
    datasets = [[], [], []]

    sorted_filenames = []
    for input_filename in input_files:
        base_filename = os.path.split(input_filename)[1]
        filenum = int(os.path.splitext(base_filename)[0].split("_")[1])
        sorted_filenames.append((filenum, input_filename))
    sorted_filenames.sort()

    for filenum, filename in sorted_filenames:
        with open(filename, errors='ignore', encoding="gb2312") as fin:
            text = fin.read()
            text = id_re.sub(r'<S ID="\1">', text)
            text = text.replace("&", "&amp;")

        try:
            xml_root = ET.fromstring(text)
        except Exception as e:
            raise RuntimeError("Cannot xml process %s" % filename) from e
        trees = [x for x in collect_trees(xml_root)]
        trees = [x[0] for x in trees if filenum != 414 or x[1] != "4366"]

        trees = "\n".join(trees)
        trees = tree_reader.read_trees(trees)
        trees = [t.prune_none().simplify_labels() for t in trees]

        assert len(trees) > 0, "No trees in %s" % filename

        shard = filenum_to_shard(filenum)
        datasets[shard].extend(trees)


    write_dataset(datasets, output_dir, dataset_name)
