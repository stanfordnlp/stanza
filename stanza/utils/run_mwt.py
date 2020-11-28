
import logging
import math

from stanza.models import mwt_expander
from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL
from stanza.utils.training import common
from stanza.utils.training.common import Mode

from stanza.utils.max_mwt_length import max_mwt_length

logger = logging.getLogger('stanza')

def check_mwt(filename):
    doc = Document(CoNLL.conll2dict(filename))
    data = doc.get_mwt_expansions(False)
    return len(data) > 0

def run_treebank(mode, paths, treebank, short_name, extra_args):
    short_language = short_name.split("_")[0]

    tokenize_dir     = paths["TOKENIZE_DATA_DIR"]
    mwt_dir          = paths["MWT_DATA_DIR"]

    train_file       = f"{mwt_dir}/{short_name}.train.in.conllu"
    dev_in_file      = f"{mwt_dir}/{short_name}.dev.in.conllu"
    dev_gold_file    = f"{mwt_dir}/{short_name}.dev.gold.conllu"
    dev_output_file  = f"{mwt_dir}/{short_name}.dev.pred.conllu"
    test_in_file     = f"{mwt_dir}/{short_name}.test.in.conllu"
    test_gold_file   = f"{mwt_dir}/{short_name}.test.gold.conllu"
    test_output_file = f"{mwt_dir}/{short_name}.test.pred.conllu"

    train_json       = f"{tokenize_dir}/{short_name}-ud-train-mwt.json"
    dev_json         = f"{tokenize_dir}/{short_name}-ud-dev-mwt.json"
    test_json        = f"{tokenize_dir}/{short_name}-ud-test-mwt.json"

    if not check_mwt(train_file):
        logger.info("No training MWTS found for %s.  Skipping" % treebank)
        return
    
    if not check_mwt(dev_in_file):
        logger.warning("No dev MWTS found for %s.  Skipping" % treebank)
        return

    if mode == Mode.TRAIN:
        max_mwt_len = math.ceil(max_mwt_length([train_json, dev_json]) * 1.1 + 1)
        logger.info("Max len: %f" % max_mwt_len)
        train_args = ['--train_file', train_file,
                      '--eval_file', dev_in_file,
                      '--output_file', dev_output_file,
                      '--gold_file', dev_gold_file,
                      '--lang', short_language,
                      '--shorthand', short_name,
                      '--mode', 'train',
                      '--max_dec_len', str(max_mwt_len)]
        train_args = train_args + extra_args
        logger.info("Running train step with args: {}".format(train_args))
        mwt_expander.main(train_args)
   
def main():
    common.main(run_treebank, "mwt", "mwt_expander")

if __name__ == "__main__":
    main()

