"""
Runs experiments with CorefModel.

Try 'python wl_coref.py -h' for more details.

Code based on

https://github.com/KarelDO/wl-coref/tree/master
https://arxiv.org/abs/2310.06165

This was a fork of

https://github.com/vdobrovolskii/wl-coref
https://aclanthology.org/2021.emnlp-main.605/

If you use Stanza's coref module in your work, please cite the following:

@misc{doosterlinck2023cawcoref,
  title={CAW-coref: Conjunction-Aware Word-level Coreference Resolution},
  author={Karel D'Oosterlinck and Semere Kiros Bitew and Brandon Papineau and Christopher Potts and Thomas Demeester and Chris Develder},
  year={2023},
  eprint={2310.06165},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url = "https://arxiv.org/abs/2310.06165",
}

@inproceedings{dobrovolskii-2021-word,
  title = "Word-Level Coreference Resolution",
  author = "Dobrovolskii, Vladimir",
  booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
  month = nov,
  year = "2021",
  address = "Online and Punta Cana, Dominican Republic",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.emnlp-main.605",
  pages = "7670--7675"
}
"""

import argparse
from contextlib import contextmanager
import datetime
import logging
import random
import sys
import dataclasses
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
    argparser.add_argument('--wandb', action='store_true', help='Start a wandb session and write the results of training.  Only applies to training.  Use --wandb_name instead to specify a name', default=False)
    argparser.add_argument('--wandb_name', default=None, help='Name of a wandb session to start when training.  Will default to the dataset short name')

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
    # if wandb, generate wandb configuration 
    if args.mode == "train":
        if args.wandb:
            import wandb
            wandb_name = args.wandb_name if args.wandb_name else f"wl_coref_{args.experiment}"
            wandb.init(name=wandb_name, config=dataclasses.asdict(config), project="stanza")
            wandb.run.define_metric('train_c_loss', summary='min')
            wandb.run.define_metric('train_s_loss', summary='min')
            wandb.run.define_metric('dev_score', summary='max')

        model = CorefModel(config=config)
        if args.weights is not None or args.warm_start:
            model.load_weights(path=args.weights, map_location="cpu",
                               noexception=args.warm_start)
        with output_running_time():
            model.train(args.wandb)
    else:
        config_update = {
            'log_norms': args.log_norms if args.log_norms is not None else False
        }
        model = CorefModel.load_model(path=args.weights, map_location="cpu",
                                      ignore={"bert_optimizer", "general_optimizer",
                                              "bert_scheduler", "general_scheduler"},
                                      config_update=config_update)
        results = model.evaluate(data_split=args.data_split,
                                 word_level_conll=args.word_level)
        logger.info(results)
