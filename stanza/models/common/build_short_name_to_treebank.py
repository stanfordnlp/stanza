import glob
import os

from stanza.models.common.constant import treebank_to_short_name
from stanza.utils import default_paths

paths = default_paths.get_default_paths()
udbase = paths["UDBASE"]

directories = glob.glob(udbase + "/UD_*")
directories.sort()

output_name = os.path.join(os.path.split(__file__)[0], "short_name_to_treebank.py")
ud_names = [os.path.split(ud_path)[1] for ud_path in directories]
short_names = [treebank_to_short_name(ud_name) for ud_name in ud_names]
max_len = max(len(x) for x in short_names) + 8
line_format = "    %-" + str(max_len) + "s '%s',\n"

print("Writing to %s" % output_name)
with open(output_name, "w") as fout:
    fout.write("# This module is autogenerated by build_short_name_to_treebank.py\n")
    fout.write("# Please do not edit\n")
    fout.write("\n")
    fout.write("SHORT_NAMES = {\n")
    for short_name, ud_name in zip(short_names, ud_names):
        fout.write(line_format % ("'" + short_name + "':", ud_name))

        if short_name.startswith("zh_"):
            short_name = "zh-hans_" + short_name[3:]
            fout.write(line_format % ("'" + short_name + "':", ud_name))
        elif short_name == 'nb_bokmaal':
            short_name = 'no_bokmaal'
            fout.write(line_format % ("'" + short_name + "':", ud_name))

    fout.write("}\n")

    fout.write("""

def short_name_to_treebank(short_name):
    return SHORT_NAMES[short_name]


""")

    max_len = max(len(x) for x in ud_names) + 5
    line_format = "    %-" + str(max_len) + "s '%s',\n"
    fout.write("CANONICAL_NAMES = {\n")
    for ud_name in ud_names:
        fout.write(line_format % ("'" + ud_name.lower() + "':", ud_name))
    fout.write("}\n")
    fout.write("""

def canonical_treebank_name(ud_name):
    if ud_name in SHORT_NAMES:
        return SHORT_NAMES[ud_name]
    return CANONICAL_NAMES.get(ud_name.lower(), ud_name)
""")
