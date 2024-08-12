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
import os
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
                           help="Adjust to override the config value of anaphoricity "
                                "batch size if you are experiencing out-of-memory "
                                "issues")
    argparser.add_argument("--disable_singletons", action="store_true",
                           help="don't predict singletons")
    argparser.add_argument("--full_pairwise", action="store_true",
                           help="use speaker and document embeddings")
    argparser.add_argument("--hidden_size", type=int,
                           help="Adjust the anaphoricity scorer hidden size")
    argparser.add_argument("--rough_k", type=int,
                           help="Adjust the number of dummies to keep")
    argparser.add_argument("--n_hidden_layers", type=int,
                           help="Adjust the anaphoricity scorer hidden layers")
    argparser.add_argument("--dummy_mix", type=float,
                           help="Adjust the dummy mix")
    argparser.add_argument("--bert_finetune_begin_epoch", type=float,
                           help="Adjust the bert finetune begin epoch")
    argparser.add_argument("--warm_start", action="store_true",
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
    argparser.add_argument("--learning_rate", default=None, type=float,
                           help="If set, update the learning rate for the model")
    argparser.add_argument("--bert_learning_rate", default=None, type=float,
                           help="If set, update the learning rate for the transformer")
    argparser.add_argument("--save_dir", default=None,
                           help="If set, update the save directory for writing models")
    argparser.add_argument("--save_name", default=None,
                           help="If set, update the save name for writing models (otherwise, section name)")
    argparser.add_argument("--score_lang", default=None,
                           help="only score a particular language for eval")
    argparser.add_argument("--log_norms", action="store_true", default=None,
                           help="If set, log all of the trainable norms each epoch.  Very noisy!")
    argparser.add_argument("--seed", type=int, default=2020,
                           help="Random seed to set")

    argparser.add_argument("--train_data", default=None, help="File to use for train data")
    argparser.add_argument("--dev_data", default=None, help="File to use for dev data")
    argparser.add_argument("--test_data", default=None, help="File to use for test data")

    argparser.add_argument('--wandb', action='store_true', help='Start a wandb session and write the results of training.  Only applies to training.  Use --wandb_name instead to specify a name', default=False)
    argparser.add_argument('--wandb_name', default=None, help='Name of a wandb session to start when training.  Will default to the dataset short name')

    args = argparser.parse_args()

    if args.warm_start and args.weights is not None:
        raise ValueError("The following options are incompatible: '--warm_start' and '--weights'")

    set_random_seed(args.seed)
    deterministic()
    config = CorefModel._load_config(args.config_file, args.experiment)
    if args.batch_size:
        config.a_scoring_batch_size = args.batch_size
    if args.hidden_size:
        config.hidden_size = args.hidden_size
    if args.n_hidden_layers:
        config.n_hidden_layers = args.n_hidden_layers
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.bert_learning_rate is not None:
        config.bert_learning_rate = args.bert_learning_rate
    if args.bert_finetune_begin_epoch is not None:
        config.bert_finetune_begin_epoch = args.bert_finetune_begin_epoch
    if args.dummy_mix is not None:
        config.dummy_mix = args.dummy_mix

    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.save_name:
        config.save_name = args.save_name
    else:
        config.save_name = args.experiment

    if args.rough_k is not None:
        config.rough_k = args.rough_k
    if args.log_norms is not None:
        config.log_norms = args.log_norms
    if args.full_pairwise:
        config.full_pairwise = args.full_pairwise
    if args.disable_singletons:
        config.singletons = False
    if args.train_data:
        config.train_data = args.train_data
    if args.dev_data:
        config.dev_data = args.dev_data
    if args.test_data:
        config.test_data = args.test_data

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
        if args.test_data:
            config_update['test_data'] = args.test_data

        if args.weights is None and config.save_name is not None:
            args.weights = config.save_name
        if not os.path.exists(args.weights) and os.path.exists(args.weights + ".pt"):
            args.weights = args.weights + ".pt"
        elif not os.path.exists(args.weights) and config.save_dir and os.path.exists(os.path.join(config.save_dir, args.weights)):
            args.weights = os.path.join(config.save_dir, args.weights)
        elif not os.path.exists(args.weights) and config.save_dir and os.path.exists(os.path.join(config.save_dir, args.weights + ".pt")):
            args.weights = os.path.join(config.save_dir, args.weights + ".pt")
        model = CorefModel.load_model(path=args.weights, map_location="cpu",
                                      ignore={"bert_optimizer", "general_optimizer",
                                              "bert_scheduler", "general_scheduler"},
                                      config_update=config_update)
        results = model.evaluate(data_split=args.data_split,
                                 word_level_conll=args.word_level, 
                                 eval_lang=args.score_lang)
        # logger.info(("mean loss", "))
        print("\t".join([str(round(i, 3)) for i in results]))
