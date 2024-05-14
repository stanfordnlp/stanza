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
        Model arguments:
        {
        batch_size (int): size of data batches used during training, 
        enc_hidden_dim (int): Size of encoder hidden state,
        enc_num_layers (int): Number of layers in the encoder LSTM,
        dec_hidden_dim (int): Size of decoder hidden state,
        dec_num_layers (int): Number of layers in the decoder LSTM,
        pgen (bool): Whether to use the pointergen feature in the model,
        coverage (bool): Whether to include coverage vectors in the decoder,
        }

        embedding_file (str): Path to the word vector pretrain file for embedding layer
        lr (float): Learning rate during training
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
        # TODO: write this out
        # Load model in
        pass 


def parse_args():
    parser = argparse.ArgumentParser()
    # Model args
    parser.add_argument("--enc_hidden_dim", type=int, default=DEFAULT_ENCODER_HIDDEN_DIM, help="Size of encoder hidden states")
    parser.add_argument("--enc_num_layers", type=int, default="Number of layers in the encoder LSTM")
    parser.add_argument("--dec_hidden_dim", type=int, default=DEFAULT_DECODER_HIDDEN_DIM, help="Size of decoder hidden state vector")
    parser.add_argument("--dec_num_layers", type=int, default=DEFAULT_DECODER_NUM_LAYERS, help="Number of layers in the decoder LSTM")
    parser.add_argument("--pgen", action="store_true", dest="pgen", default=False, help="Use pointergen probabilities to point to input text")
    parser.add_argument("--coverage", action="store_true", dest="coverage", default=False, help="Use coverage vectors during decoding stage")
    # Training args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for data processing")
    parser.add_argument("--save_name", type="str", default="", help="Path to destination for final trained model.")
    parser.add_argument("--eval_file", type="str", default="", help="Path to the validation set file")
    parser.add_argument("--train_file", type="str", default="", help="Path to the training data file")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wordvec_pretrain_file", type="str", default="", help="Path to pretrained word embeddings file")

    # TODO set default values that make sense for the path vars
    return parser

def main():
    argparser = parse_args()
    args = argparser.parse_args()

    enc_hidden_dim = args.enc_hidden_dim
    enc_num_layers = args.enc_num_layers
    dec_hidden_dim = args.dec_hidden_dim
    dec_num_layers = args.dec_num_layers
    pgen = args.pgen
    coverage = args.pgen

    batch_size = args.batch_size
    save_name = args.save_name
    eval_file = args.eval_file
    train_file = args.train_file
    num_epochs = args.num_epochs
    lr = args.lr
    wordvec_pretrain_file = args.wordvec_pretrain_file

    # TODO: Check files for existence and raise Exceptions accordingly

    args = vars(args)

    trainer = SummarizationTrainer(
        model_args=args,
        embedding_file=wordvec_pretrain_file,
        lr=lr
    )
    trainer.train(
        num_epochs=num_epochs,
        save_name=save_name,
        train_file=train_file,
        eval_file=eval_file
    )


if __name__ == "__main__":
    main()