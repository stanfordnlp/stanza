
import argparse
import glob
import logging
import os
import re
import subprocess
import sys

from stanza.models.common.short_name_to_treebank import canonical_treebank_name
import stanza.utils.datasets.prepare_tokenizer_data as prepare_tokenizer_data
import stanza.utils.default_paths as default_paths

logger = logging.getLogger('stanza')

# RE to see if the index of a conllu line represents an MWT
MWT_RE = re.compile("^[0-9]+[-][0-9]+")

# RE to see if the index of a conllu line represents an MWT or copy node
MWT_OR_COPY_RE = re.compile("^[0-9]+[-.][0-9]+")

# more restrictive than an actual int as we expect certain formats in the conllu files
INT_RE = re.compile("^[0-9]+$")

CONLLU_TO_TXT_PERL = os.path.join(os.path.split(__file__)[0], "conllu_to_text.pl")

def convert_conllu_to_txt(tokenizer_dir, short_name, shards=("train", "dev", "test")):
    """
    Uses the udtools perl script to convert a conllu file to txt

    TODO: switch to a python version to get rid of some perl dependence
    """
    for dataset in shards:
        output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
        output_txt = f"{tokenizer_dir}/{short_name}.{dataset}.txt"

        if not os.path.exists(output_conllu):
            # the perl script doesn't raise an error code for file not found!
            raise FileNotFoundError("Cannot convert %s as the file cannot be found" % output_conllu)
        # use an external script to produce the txt files
        subprocess.check_output(f"perl {CONLLU_TO_TXT_PERL} {output_conllu} > {output_txt}", shell=True)

def mwt_name(base_dir, short_name, dataset):
    return os.path.join(base_dir, f"{short_name}-ud-{dataset}-mwt.json")

def tokenizer_conllu_name(base_dir, short_name, dataset):
    return os.path.join(base_dir, f"{short_name}.{dataset}.gold.conllu")

def prepare_tokenizer_dataset_labels(input_txt, input_conllu, tokenizer_dir, short_name, dataset):
    prepare_tokenizer_data.main([input_txt,
                                 input_conllu,
                                 "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                 "-m", mwt_name(tokenizer_dir, short_name, dataset)])

def prepare_tokenizer_treebank_labels(tokenizer_dir, short_name):
    """
    Given the txt and gold.conllu files, prepare mwt and label files for train/dev/test
    """
    for dataset in ("train", "dev", "test"):
        output_txt = f"{tokenizer_dir}/{short_name}.{dataset}.txt"
        output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
        try:
            prepare_tokenizer_dataset_labels(output_txt, output_conllu, tokenizer_dir, short_name, dataset)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print("Failed to convert %s to %s" % (output_txt, output_conllu))
            raise

def read_sentences_from_conllu(filename):
    sents = []
    cache = []
    with open(filename, encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if len(line) == 0:
                if len(cache) > 0:
                    sents.append(cache)
                    cache = []
                continue
            cache.append(line)
        if len(cache) > 0:
            sents.append(cache)
    return sents

def maybe_add_fake_dependencies(lines):
    """
    Possibly add fake dependencies in columns 6 and 7 (counting from 0)

    The conllu scripts need the dependencies column filled out, so in
    the case of models we build without dependency data, we need to
    add those fake dependencies in order to use the eval script etc
    """
    new_lines = []
    root_idx = None
    first_idx = None
    for line_idx, line in enumerate(lines):
        if line.startswith("#"):
            new_lines.append(line)
            continue

        pieces = line.split("\t")
        if MWT_OR_COPY_RE.match(pieces[0]):
            new_lines.append(line)
            continue

        token_idx = int(pieces[0])
        if pieces[6] != '_':
            if pieces[6] == '0':
                root_idx = token_idx
            new_lines.append(line)
        elif token_idx == 1:
            # note that the comments might make this not the first line
            # we keep track of this separately so we can either make this the root,
            # or set this to be the root later
            first_idx = line_idx
            new_lines.append(pieces)
        else:
            pieces[6] = "1"
            pieces[7] = "dep"
            new_lines.append("\t".join(pieces))
    if first_idx is not None:
        if root_idx is None:
            new_lines[first_idx][6] = "0"
            new_lines[first_idx][7] = "root"
        else:
            new_lines[first_idx][6] = str(root_idx)
            new_lines[first_idx][7] = "dep"
        new_lines[first_idx] = "\t".join(new_lines[first_idx])
    return new_lines

def write_sentences_to_conllu(filename, sents):
    with open(filename, 'w', encoding="utf-8") as outfile:
        for lines in sents:
            lines = maybe_add_fake_dependencies(lines)
            for line in lines:
                print(line, file=outfile)
            print("", file=outfile)

def find_treebank_dataset_file(treebank, udbase_dir, dataset, extension, fail=False):
    """
    For a given treebank, dataset, extension, look for the exact filename to use.

    Sometimes the short name we use is different from the short name
    used by UD.  For example, Norwegian or Chinese.  Hence the reason
    to not hardcode it based on treebank

    set fail=True to fail if the file is not found
    """
    if treebank.startswith("UD_Korean") and treebank.endswith("_seg"):
        treebank = treebank[:-4]
    filename = os.path.join(udbase_dir, treebank, f"*-ud-{dataset}.{extension}")
    files = glob.glob(filename)
    if len(files) == 0:
        if fail:
            raise FileNotFoundError("Could not find any treebank files which matched {}".format(filename))
        else:
            return None
    elif len(files) == 1:
        return files[0]
    else:
        raise RuntimeError(f"Unexpected number of files matched '{udbase_dir}/{treebank}/*-ud-{dataset}.{extension}'")

def mostly_underscores(filename):
    """
    Certain treebanks have proprietary data, so the text is hidden

    For example:
      UD_Arabic-NYUAD
      UD_English-ESL
      UD_English-GUMReddit
      UD_Hindi_English-HIENCS
      UD_Japanese-BCCWJ
    """
    underscore_count = 0
    total_count = 0
    for line in open(filename).readlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        total_count = total_count + 1
        pieces = line.split("\t")
        if pieces[1] in ("_", "-"):
            underscore_count = underscore_count + 1
    return underscore_count / total_count > 0.5

def num_words_in_file(conllu_file):
    """
    Count the number of non-blank lines in a conllu file
    """
    count = 0
    with open(conllu_file) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            count = count + 1
    return count


def get_ud_treebanks(udbase_dir, filtered=True):
    """
    Looks in udbase_dir for all the treebanks which have both train, dev, and test
    """
    treebanks = sorted(glob.glob(udbase_dir + "/UD_*"))
    # skip UD_English-GUMReddit as it is usually incorporated into UD_English-GUM
    treebanks = [os.path.split(t)[1] for t in treebanks]
    treebanks = [t for t in treebanks if t != "UD_English-GUMReddit"]
    if filtered:
        treebanks = [t for t in treebanks
                     if (find_treebank_dataset_file(t, udbase_dir, "train", "conllu") and
                         # this will be fixed using XV
                         #find_treebank_dataset_file(t, udbase_dir, "dev", "conllu") and
                         find_treebank_dataset_file(t, udbase_dir, "test", "conllu"))]
        treebanks = [t for t in treebanks
                     if not mostly_underscores(find_treebank_dataset_file(t, udbase_dir, "train", "conllu"))]
        # eliminate partial treebanks (fixed with XV) for which we only have 1000 words or less
        # if the train set is small and the test set is large enough, we'll flip them
        treebanks = [t for t in treebanks
                     if (find_treebank_dataset_file(t, udbase_dir, "dev", "conllu") or
                         num_words_in_file(find_treebank_dataset_file(t, udbase_dir, "train", "conllu")) > 1000 or
                         num_words_in_file(find_treebank_dataset_file(t, udbase_dir, "test", "conllu")) > 5000)]
    return treebanks

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('treebanks', type=str, nargs='+', help='Which treebanks to run on.  Use all_ud or ud_all for all UD treebanks')

    return parser


def main(process_treebank, add_specific_args=None):
    logger.info("Datasets program called with:\n" + " ".join(sys.argv))

    parser = build_argparse()
    if add_specific_args is not None:
        add_specific_args(parser)
    args = parser.parse_args()

    paths = default_paths.get_default_paths()

    treebanks = []
    for treebank in args.treebanks:
        if treebank.lower() in ('ud_all', 'all_ud'):
            ud_treebanks = get_ud_treebanks(paths["UDBASE"])
            treebanks.extend(ud_treebanks)
        else:
            # If this is a known UD short name, use the official name (we need it for the paths)
            treebank = canonical_treebank_name(treebank)
            treebanks.append(treebank)

    for treebank in treebanks:
        process_treebank(treebank, paths, args)
