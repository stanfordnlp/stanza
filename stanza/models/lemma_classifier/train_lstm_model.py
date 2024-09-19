"""
The code in this file works to train a lemma classifier for 's
"""

import argparse
import logging
import os

import torch
import torch.nn as nn

from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.lemma_classifier.base_trainer import BaseLemmaClassifierTrainer
from stanza.models.lemma_classifier.constants import DEFAULT_BATCH_SIZE
from stanza.models.lemma_classifier.lstm_model import LemmaClassifierLSTM

logger = logging.getLogger('stanza.lemmaclassifier')

class LemmaClassifierTrainer(BaseLemmaClassifierTrainer):
    """
    Class to assist with training a LemmaClassifierLSTM
    """

    def __init__(self, model_args: dict, embedding_file: str, use_charlm: bool = False, charlm_forward_file: str = None, charlm_backward_file: str = None, lr: float = 0.001, loss_func: str = None):
        """
        Initializes the LemmaClassifierTrainer class.

        Args:
            model_args (dict): Various model shape parameters
            embedding_file (str): What word embeddings file to use.  Use a Stanza pretrain .pt
            use_charlm (bool, optional): Whether to use charlm embeddings as well. Defaults to False.
            charlm_forward_file (str): Path to the forward pass embeddings for the charlm
            charlm_backward_file (str): Path to the backward pass embeddings for the charlm
            upos_emb_dim (int): The dimension size of UPOS tag embeddings
            num_heads (int): The number of attention heads to use.
            lr (float): Learning rate, defaults to 0.001.
            loss_func (str): Which loss function to use (either 'ce' or 'weighted_bce')

        Raises:
            FileNotFoundError: If the forward charlm file is not present
            FileNotFoundError: If the backward charlm file is not present
        """
        super().__init__()

        self.model_args = model_args

        # Load word embeddings
        pt = load_pretrain(embedding_file)
        self.pt_embedding = pt

        # Load CharLM embeddings
        if use_charlm and charlm_forward_file is not None and not os.path.exists(charlm_forward_file):
            raise FileNotFoundError(f"Could not find forward charlm file: {charlm_forward_file}")
        if use_charlm and charlm_backward_file is not None and not os.path.exists(charlm_backward_file):
            raise FileNotFoundError(f"Could not find backward charlm file: {charlm_backward_file}")

        # TODO: just pass around the args instead
        self.use_charlm = use_charlm
        self.charlm_forward_file = charlm_forward_file
        self.charlm_backward_file = charlm_backward_file
        self.lr = lr

        # Find loss function
        if loss_func == "ce":
            self.criterion = nn.CrossEntropyLoss()
            self.weighted_loss = False
            logger.debug("Using CE loss")
        elif loss_func == "weighted_bce":
            self.criterion = nn.BCEWithLogitsLoss()
            self.weighted_loss = True  # used to add weights during train time.
            logger.debug("Using Weighted BCE loss")
        else:
            raise ValueError("Must enter a valid loss function (e.g. 'ce' or 'weighted_bce')")

    def build_model(self, label_decoder, upos_to_id, known_words, target_words, target_upos):
        return LemmaClassifierLSTM(self.model_args, self.output_dim, self.pt_embedding, label_decoder, upos_to_id, known_words, target_words, target_upos,
                                   use_charlm=self.use_charlm, charlm_forward_file=self.charlm_forward_file, charlm_backward_file=self.charlm_backward_file)

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=256, help="Size of hidden layer")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=os.path.join(os.path.dirname(__file__), "pretrain", "glove.pt"), help='Exact name of the pretrain file to read')
    parser.add_argument("--charlm", action='store_true', dest='use_charlm', default=False, help="Whether not to use the charlm embeddings")
    parser.add_argument('--charlm_shorthand', type=str, default=None, help="Shorthand for character-level language model training corpus.")
    parser.add_argument("--charlm_forward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_forward.pt"), help="Path to forward charlm file")
    parser.add_argument("--charlm_backward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_backwards.pt"), help="Path to backward charlm file")
    parser.add_argument("--upos_emb_dim", type=int, default=20, help="Dimension size for UPOS tag embeddings.")
    parser.add_argument("--use_attn", action='store_true', dest='attn', default=False, help='Whether to use multihead attention instead of LSTM.')
    parser.add_argument("--num_heads", type=int, default=0, help="Number of heads to use for multihead attention.")
    parser.add_argument("--save_name", type=str, default=os.path.join(os.path.dirname(__file__), "saved_models", "lemma_classifier_model_weighted_loss_charlm_new.pt"), help="Path to model save file")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of examples to include in each batch")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(__file__), "data", "processed_ud_en", "combined_train.txt"), help="Full path to training file")
    parser.add_argument("--weighted_loss", action='store_true', dest='weighted_loss', default=False, help="Whether to use weighted loss during training.")
    parser.add_argument("--eval_file", type=str, default=os.path.join(os.path.dirname(__file__), "data", "processed_ud_en", "combined_dev.txt"), help="Path to dev file used to evaluate model for saves")
    parser.add_argument("--force", action='store_true', default=False, help='Whether or not to clobber an existing save file')
    return parser

def main(args=None, predefined_args=None):
    parser = build_argparse()
    args = parser.parse_args(args) if predefined_args is None else predefined_args

    wordvec_pretrain_file = args.wordvec_pretrain_file
    use_charlm = args.use_charlm
    charlm_forward_file = args.charlm_forward_file
    charlm_backward_file = args.charlm_backward_file
    upos_emb_dim = args.upos_emb_dim
    use_attention = args.attn
    num_heads = args.num_heads
    save_name = args.save_name
    lr = args.lr
    num_epochs = args.num_epochs
    train_file = args.train_file
    weighted_loss = args.weighted_loss
    eval_file = args.eval_file

    args = vars(args)

    if os.path.exists(save_name) and not args.get('force', False):
        raise FileExistsError(f"Save name {save_name} already exists. Training would override existing data. Aborting...")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file {train_file} not found. Try again with a valid path.")

    logger.info("Running training script with the following args:")
    for arg in args:
        logger.info(f"{arg}: {args[arg]}")
    logger.info("------------------------------------------------------------")

    trainer = LemmaClassifierTrainer(model_args=args,
                                     embedding_file=wordvec_pretrain_file,
                                     use_charlm=use_charlm,
                                     charlm_forward_file=charlm_forward_file,
                                     charlm_backward_file=charlm_backward_file,
                                     lr=lr,
                                     loss_func="weighted_bce" if weighted_loss else "ce",
                                     )

    trainer.train(
        num_epochs=num_epochs, save_name=save_name, args=args, eval_file=eval_file, train_file=train_file
    )

    return trainer

if __name__ == "__main__":
    main()

