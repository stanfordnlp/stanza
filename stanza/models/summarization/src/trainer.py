import argparse
import logging
import os 
import torch
import sys
import torch.nn as nn
import torch.optim as optim

# To add Stanza modules, TODO remove this and just EXPORT this to the sys path manually before running
ROOT = '/Users/alexshan/Desktop/stanza'
sys.path.append(ROOT)

from stanza.models.common.utils import default_device
from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.summarization.constants import * 
from stanza.models.summarization.src.model import *
from stanza.utils.get_tqdm import get_tqdm

from typing import List, Tuple, Any, Mapping

torch.set_printoptions(threshold=100, edgeitems=5, linewidth=100)
logger = logging.getLogger('stanza.summarization') 
logger.propagate = False

# Check if the logger has handlers already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

tqdm = get_tqdm()


class SummarizationTrainer():

    def __init__(self, model_args: dict, embedding_file: str, lr: float):
        """
        TODO: finish this documentation

        Model arguments:
        {
        batch size, 
        encoder hidden dim,
        encoder num layers,
        decoder hidden dim,
        decoder num layers,
        pgen,
        coverage,
        }

        embedding_file:

        lr:
        """
        self.model_args = model_args

        pt = load_pretrain(embedding_file)
        self.pt_embedding = pt
        self.lr = lr 

    def build_model(self) -> BaselineSeq2Seq:
        """
        Build the model for training using the model args

        Raises any errors depending on model argument errors
        """

        # parse input for valid args
        batch_size = self.model_args.get("batch_size", DEFAULT_BATCH_SIZE)
        encoder_hidden_dim = self.model_args.get("encoder_hidden_dim", DEFAULT_ENCODER_HIDDEN_DIM)
        encoder_num_layers = self.model_args.get("encoder_num_layers", DEFAULT_ENCODER_NUM_LAYERS)
        decoder_hidden_dim = self.model_args.get("decoder_hidden_dim", DEFAULT_DECODER_HIDDEN_DIM)
        decoder_num_layers = self.model_args.get("decoder_num_layers", DEFAULT_DECODER_NUM_LAYERS)
        pgen = self.model_args.get("pgen", False)
        coverage = self.model_args.get("coverage", False)

        parsed_model_args = {
            "batch_size": batch_size,
            "encoder_hidden_dim": encoder_hidden_dim,
            "encoder_num_layers": encoder_num_layers,
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_num_layers": decoder_num_layers,
            "pgen": pgen,
            "coverage": coverage
        }

        # return the model obj
        return BaselineSeq2Seq(parsed_model_args, self.pt_embedding)

    def train(self, num_epochs: int, save_name: str, train_file: str, eval_file: str) -> None:
        """
        Trains a model on batches of texts

        Args:
            num_epochs (int): Number of training epochs 
            save_name (str): Path to store trained model
            eval_file (str): Path to the validation set file for evaluating model checkpoints
            train_file (str): Path to training data file containing tokenized text for each article

        Returns:
            None (model with best validation set performance will be saved to the save file)
        """
        pass 

def main():
    # TODO: parse cli args, build the model args mapping


    trainer = SummarizationTrainer()
    trainer.train()


if __name__ == "__main__":
    main()