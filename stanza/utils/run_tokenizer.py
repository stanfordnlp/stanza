"""
This script allows for training or testing on dev / test of the UD tokenizer.

If run with a single treebank name, it will train or test that treebank.
If run with ud_all or all_ud, it will iterate over all UD treebanks it can find.

Args are given as follows:

python run_tokenizer.py [mode] <treebank> [tokenizer args...]

The first argument, mode, is optional.  It can either be --train (or
nothing) to train a model and report the dev score, --score_dev to
just report the dev score, or --score_test to report the test score.

After specifying the treebank, any further arguments will be passed to the tokenizer.
"""

import math
import os
import subprocess
import sys

from enum import Enum

from stanza.models.common.constant import treebank_to_short_name
from stanza.models import tokenizer
from stanza.utils.avg_sent_len import avg_sent_len
import stanza.utils.default_paths as default_paths
import prepare_tokenizer_treebank

class Mode(Enum):
    TRAIN = 1
    SCORE_DEV = 2
    SCORE_TEST = 3

def run_eval_script(eval_gold, eval_pred):
    # TODO: this is a silly way of doing this
    # but the eval script expects sys args and prints the results to stdout
    eval_script = os.path.join(os.path.split(__file__)[0], "conll18_ud_eval.py")
    results = subprocess.check_output([eval_script, "-v", eval_gold, eval_pred])
    results = results.decode(encoding="utf-8")
    results = [x.split("|")[3].strip() for x in results.split("\n")[2:5]]
    return " ".join(results)
    

def run_treebank(mode, paths, treebank, short_name, extra_args):
    tokenize_dir = paths["TOKENIZE_DATA_DIR"]

    short_language = short_name.split("_")[0]
    if short_language == 'vi':
        label_type = "--json_file"
        label_file = f"{tokenize_dir}/{short_name}-ud-train.json"
        dev_type = "--json_file"
        dev_file = f"{tokenize_dir}/{short_name}-ud-dev.json"
        test_type = "--json_file"
        test_file = f"{tokenize_dir}/{short_name}-ud-test.json"
        train_type = "--txt_file"
        train_file = f"{tokenize_dir}/{short_name}.train.txt"
        train_dev_args = ["--dev_json_file", dev_file]
    else:
        label_type = "--label_file"
        label_file = f"{tokenize_dir}/{short_name}-ud-train.toklabels"
        dev_type = "--txt_file"
        dev_file = f"{tokenize_dir}/{short_name}.dev.txt"
        test_type = "--txt_file"
        test_file = f"{tokenize_dir}/{short_name}.test.txt"
        train_type = "--txt_file"
        train_file = f"{tokenize_dir}/{short_name}.train.txt"
        train_dev_args = ["--dev_txt_file", dev_file, "--dev_label_file", f"{tokenize_dir}/{short_name}-ud-dev.toklabels"]
    
    if short_language == "zh":
        extra_args = ["--skip_newline"] + extra_args

    dev_gold = f"{tokenize_dir}/{short_name}.dev.gold.conllu"
    test_gold = f"{tokenize_dir}/{short_name}.test.gold.conllu"

    dev_mwt = f"{tokenize_dir}/{short_name}-ud-dev-mwt.json"
    test_mwt = f"{tokenize_dir}/{short_name}-ud-test-mwt.json"

    # TODO: use a tmp file for this?
    dev_pred = f"{tokenize_dir}/{short_name}.dev.pred.conllu"
    test_pred = f"{tokenize_dir}/{short_name}.test.pred.conllu"

    if mode == Mode.TRAIN:
        seqlen = str(math.ceil(avg_sent_len(label_file) * 3 / 100) * 100)
        train_args = ([label_type, label_file, train_type, train_file, "--lang", short_language,
                       "--max_seqlen", seqlen, "--mwt_json_file", dev_mwt] +
                      train_dev_args +
                      ["--dev_conll_gold", dev_gold, "--conll_file", dev_pred, "--shorthand", short_name] +
                      extra_args)
        print("Running train step with args: {}".format(train_args))
        tokenizer.main(train_args)
    
    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ["--mode", "predict", dev_type, dev_file, "--lang", short_language,
                    "--conll_file", dev_pred, "--shorthand", short_name, "--mwt_json_file", dev_mwt]
        dev_args = dev_args + extra_args
        print("Running dev step with args: {}".format(dev_args))
        tokenizer.main(dev_args)

        # TODO: log these results?  The original script logged them to
        # echo $results $args >> ${TOKENIZE_DATA_DIR}/${short}.results

        results = run_eval_script(dev_gold, dev_pred)
        print(results)

    if mode == Mode.SCORE_TEST:
        test_args = ["--mode", "predict", test_type, test_file, "--lang", short_language,
                     "--conll_file", test_pred, "--shorthand", short_name, "--mwt_json_file", test_mwt]
        test_args = test_args + extra_args
        print("Running test step with args: {}".format(test_args))
        tokenizer.main(test_args)

        results = run_eval_script(test_gold, test_pred)
        print(results)

def main():
    paths = default_paths.get_default_paths()
    args = sys.argv[1:]
    if args[0].startswith("--"):
        mode = Mode[args[0][2:].upper()]
        args = args[1:]
    else:
        mode = Mode.TRAIN
    treebank = args[0]
    extra_args = args[1:]

    if treebank.lower() in ('ud_all', 'all_ud'):
        treebanks = prepare_tokenizer_treebank.get_ud_treebanks(paths["UDBASE"])

        for t in treebanks:
            short_name = treebank_to_short_name(t)
            if mode == Mode.TRAIN and os.path.exists("saved_models/tokenize/%s_tokenizer.pt" % short_name):
                print("echo %s: %s exists, skipping!" % (t, short_name))
                continue

            print("echo %s: %s" % (t, short_name))
            run_treebank(mode, paths, t, short_name, extra_args)
    else:
        short_name = treebank_to_short_name(treebank)
        run_treebank(mode, paths, treebank, short_name, extra_args)    

if __name__ == "__main__":
    main()
