from enum import Enum
import glob
import os
import re

import xml.etree.ElementTree as ET

from stanza.models.constituency import tree_reader
from stanza.utils.datasets.constituency.utils import write_dataset
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

class Version(Enum):
    V51   = 1
    V90   = 2

def filenum_to_shard_51(filenum):
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

def filenum_to_shard_90(filenum):
    if filenum >= 1 and filenum <= 40:
        return 2
    if filenum >= 900 and filenum <= 931:
        return 2
    if filenum in (1018, 1020, 1036, 1044, 1060, 1061, 1072, 1118, 1119, 1132, 1141, 1142, 1148):
        return 2
    if filenum >= 2165 and filenum <= 2180:
        return 2
    if filenum >= 2295 and filenum <= 2310:
        return 2
    if filenum >= 2570 and filenum <= 2602:
        return 2
    if filenum >= 2800 and filenum <= 2819:
        return 2
    if filenum >= 3110 and filenum <= 3145:
        return 2


    if filenum >= 41 and filenum <= 80:
        return 1
    if filenum >= 1120 and filenum <= 1129:
        return 1
    if filenum >= 2140 and filenum <= 2159:
        return 1
    if filenum >= 2280 and filenum <= 2294:
        return 1
    if filenum >= 2550 and filenum <= 2569:
        return 1
    if filenum >= 2775 and filenum <= 2799:
        return 1
    if filenum >= 3080 and filenum <= 3109:
        return 1

    if filenum >= 81 and filenum <= 900:
        return 0
    if filenum >= 1001 and filenum <= 1017:
        return 0
    if filenum in (1019, 1130, 1131):
        return 0
    if filenum >= 1021 and filenum <= 1035:
        return 0
    if filenum >= 1037 and filenum <= 1043:
        return 0
    if filenum >= 1045 and filenum <= 1059:
        return 0
    if filenum >= 1062 and filenum <= 1071:
        return 0
    if filenum >= 1073 and filenum <= 1117:
        return 0
    if filenum >= 1133 and filenum <= 1140:
        return 0
    if filenum >= 1143 and filenum <= 1147:
        return 0
    if filenum >= 1149 and filenum <= 2139:
        return 0
    if filenum >= 2160 and filenum <= 2164:
        return 0
    if filenum >= 2181 and filenum <= 2279:
        return 0
    if filenum >= 2311 and filenum <= 2549:
        return 0
    if filenum >= 2603 and filenum <= 2774:
        return 0
    if filenum >= 2820 and filenum <= 3079:
        return 0
    if filenum >= 4000 and filenum <= 7017:
        return 0


def collect_trees_s(root):
    if root.tag == 'S':
        yield root.text, root.attrib['ID']

    for child in root:
        for tree in collect_trees_s(child):
            yield tree

def collect_trees_text(root):
    if root.tag == 'TEXT' and len(root.text.strip()) > 0:
        yield root.text, None

    if root.tag == 'TURN' and len(root.text.strip()) > 0:
        yield root.text, None

    for child in root:
        for tree in collect_trees_text(child):
            yield tree


id_re = re.compile("<S ID=([0-9a-z]+)>")
su_re = re.compile("<(su|msg) id=([0-9a-zA-Z_=]+)>")

def convert_ctb(input_dir, output_dir, dataset_name, version):
    input_files = glob.glob(os.path.join(input_dir, "*"))

    # train, dev, test
    datasets = [[], [], []]

    sorted_filenames = []
    for input_filename in input_files:
        base_filename = os.path.split(input_filename)[1]
        filenum = int(os.path.splitext(base_filename)[0].split("_")[1])
        sorted_filenames.append((filenum, input_filename))
    sorted_filenames.sort()

    for filenum, filename in tqdm(sorted_filenames):
        if version is Version.V51:
            with open(filename, errors='ignore', encoding="gb2312") as fin:
                text = fin.read()
        else:
            with open(filename, encoding="utf-8") as fin:
                text = fin.read()
            if text.find("<TURN>") >= 0 and text.find("</TURN>") < 0:
                text = text.replace("<TURN>", "")
            if filenum in (4205, 4208, 4289):
                text = text.replace("<)", "&lt;)").replace(">)", "&gt;)")
            if filenum >= 4000 and filenum <= 4411:
                if text.find("<segment") >= 0:
                    text = text.replace("<segment id=", "<S ID=").replace("</segment>", "</S>")
                elif text.find("<seg") < 0:
                    text = "<TEXT>\n%s</TEXT>\n" % text
                else:
                    text = text.replace("<seg id=", "<S ID=").replace("</seg>", "</S>")
                text = "<foo>\n%s</foo>\n" % text
            if filenum >= 5000 and filenum <= 5558 or filenum >= 6000 and filenum <= 6700 or filenum >= 7000 and filenum <= 7017:
                text = su_re.sub("", text)
                if filenum in (6066, 6453):
                    text = text.replace("<", "&lt;").replace(">", "&gt;")
                text = "<foo><TEXT>\n%s</TEXT></foo>\n" % text
        text = id_re.sub(r'<S ID="\1">', text)
        text = text.replace("&", "&amp;")

        try:
            xml_root = ET.fromstring(text)
        except Exception as e:
            print(text[:1000])
            raise RuntimeError("Cannot xml process %s" % filename) from e
        trees = [x for x in collect_trees_s(xml_root)]
        if version is Version.V90 and len(trees) == 0:
            trees = [x for x in collect_trees_text(xml_root)]

        if version is Version.V51:
            trees = [x[0] for x in trees if filenum != 414 or x[1] != "4366"]
        else:
            trees = [x[0] for x in trees]

        trees = "\n".join(trees)
        try:
            trees = tree_reader.read_trees(trees, use_tqdm=False)
        except ValueError as e:
            print(text[:300])
            raise RuntimeError("Could not process the tree text in %s" % filename)
        trees = [t.prune_none().simplify_labels() for t in trees]

        assert len(trees) > 0, "No trees in %s" % filename

        if version is Version.V51:
            shard = filenum_to_shard_51(filenum)
        else:
            shard = filenum_to_shard_90(filenum)
        datasets[shard].extend(trees)


    write_dataset(datasets, output_dir, dataset_name)
