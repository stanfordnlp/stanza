""" Runs experiments with CorefModel.

Try 'python wl_coref.py -h' for more details.
"""

import argparse
from contextlib import contextmanager
import datetime
import logging
import random
import sys
import time

import numpy as np  # type: ignore
import torch        # type: ignore

from stanza.models.common.utils import set_random_seed
from stanza.models.coref.model import CorefModel


logger = logging.getLogger('stanza')

@contextmanager
def output_running_time():
    """ Prints the time elapsed in the context """
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        logger.info(f"Total running time: {delta}")


def deterministic() -> None:
    torch.backends.cudnn.deterministic = True   # type: ignore
    torch.backends.cudnn.benchmark = False      # type: ignore


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=("train", "eval"))
    argparser.add_argument("experiment")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--data-split", choices=("train", "dev", "test"),
                           default="test",
                           help="Data split to be used for evaluation."
                                " Defaults to 'test'."
                                " Ignored in 'train' mode.")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--warm-start", action="store_true",
                           help="If set, the training will resume from the"
                                " last checkpoint saved if any. Ignored in"
                                " evaluation modes."
                                " Incompatible with '--weights'.")
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--word-level", action="store_true",
                           help="If set, output word-level conll-formatted"
                                " files in evaluation modes. Ignored in"
                                " 'train' mode.")
    argparser.add_argument("--bert_learning_rate", default=None, type=float,
                           help="If set, update the learning rate for the transformer")
    argparser.add_argument("--save_dir", default=None,
                           help="If set, update the save directory for writing models")
    argparser.add_argument("--log_norms", action="store_true", default=None,
                           help="If set, log all of the trainable norms each epoch.  Very noisy!")
    args = argparser.parse_args()

    if args.warm_start and args.weights is not None:
        raise ValueError("The following options are incompatible: '--warm_start' and '--weights'")

    set_random_seed(2020)
    deterministic()
    config = CorefModel._load_config(args.config_file, args.experiment)
    if args.batch_size:
        config.a_scoring_batch_size = args.batch_size
    if args.bert_learning_rate is not None:
        config.bert_learning_rate = args.bert_learning_rate
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.log_norms is not None:
        config.log_norms = args.log_norms

    model = CorefModel(config=config)

    if args.mode == "train":
        if args.weights is not None or args.warm_start:
            model.load_weights(path=args.weights, map_location="cpu",
                               noexception=args.warm_start)
        with output_running_time():
            model.train()
    else:
        model.load_weights(path=args.weights, map_location="cpu",
                           ignore={"bert_optimizer", "general_optimizer",
                                   "bert_scheduler", "general_scheduler"})
        results = model.evaluate(data_split=args.data_split,
                                 word_level_conll=args.word_level)
        logger.info(results)
