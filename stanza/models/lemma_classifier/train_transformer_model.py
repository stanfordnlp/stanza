"""
This file contains code used to train a baseline transformer model to classify on a lemma of a particular token.
"""

import argparse
import os
import sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from stanza.models.lemma_classifier.base_trainer import BaseLemmaClassifierTrainer
from stanza.models.lemma_classifier.constants import DEFAULT_BATCH_SIZE
from stanza.models.lemma_classifier.transformer_model import LemmaClassifierWithTransformer
from stanza.models.common.utils import default_device

logger = logging.getLogger('stanza.lemmaclassifier')

class TransformerBaselineTrainer(BaseLemmaClassifierTrainer):
    """
    Class to assist with training a baseline transformer model to classify on token lemmas.
    To find the model spec, refer to `model.py` in this directory.
    """

    def __init__(self, model_args: dict, transformer_name: str = "roberta", loss_func: str = "ce", lr: int = 0.001):
        """
        Creates the Trainer object

        Args:
            transformer_name (str, optional): What kind of transformer to use for embeddings. Defaults to "roberta".
            loss_func (str, optional): Which loss function to use (either 'ce' or 'weighted_bce'). Defaults to "ce".
            lr (int, optional): learning rate for the optimizer. Defaults to 0.001.
        """
        super().__init__()

        self.model_args = model_args

        # Find loss function
        if loss_func == "ce":
            self.criterion = nn.CrossEntropyLoss()
            self.weighted_loss = False
        elif loss_func == "weighted_bce":
            self.criterion = nn.BCEWithLogitsLoss()
            self.weighted_loss = True  # used to add weights during train time.
        else:
            raise ValueError("Must enter a valid loss function (e.g. 'ce' or 'weighted_bce')")

        self.transformer_name = transformer_name
        self.lr = lr

    def set_layer_learning_rates(self, transformer_lr: float, mlp_lr: float) -> torch.optim:
        """
        Sets learning rates for each layer of the model.
        Currently, the model has the transformer layer and the MLP layer, so these are tweakable.

        Returns (torch.optim): An Adam optimizer with the learning rates adjusted per layer.

        Currently unused - could be refactored into the parent class's train method,
        or the parent class could call a build_optimizer and this subclass would use the optimizer
        """
        transformer_params, mlp_params = [], []
        for name, param in self.model.named_parameters():
            if 'transformer' in name:
                transformer_params.append(param)
            elif 'mlp' in name:
                mlp_params.append(param)
        optimizer = optim.Adam([
            {"params": transformer_params, "lr": transformer_lr},
            {"params": mlp_params, "lr": mlp_lr}
        ])
        return optimizer

    def build_model(self, label_decoder, upos_to_id, known_words, target_words, target_upos):
        return LemmaClassifierWithTransformer(model_args=self.model_args, output_dim=self.output_dim, transformer_name=self.transformer_name, label_decoder=label_decoder, target_words=target_words, target_upos=target_upos)


def main(args=None, predefined_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_name", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "big_model_roberta_weighted_loss.pt"), help="Path to model save file")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_sets", "combined_train.txt"), help="Full path to training file")
    parser.add_argument("--model_type", type=str, default="roberta", help="Which transformer to use ('bert' or 'roberta')")
    parser.add_argument("--bert_model", type=str, default=None, help="Use a specific transformer instead of the default bert/roberta")
    parser.add_argument("--loss_fn", type=str, default="weighted_bce", help="Which loss function to train with (e.g. 'ce' or 'weighted_bce')")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of examples to include in each batch")
    parser.add_argument("--eval_file", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_sets", "combined_dev.txt"), help="Path to dev file used to evaluate model for saves")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--force", action='store_true', default=False, help='Whether or not to clobber an existing save file')

    args = parser.parse_args(args) if predefined_args is None else predefined_args

    save_name = args.save_name
    num_epochs = args.num_epochs
    train_file = args.train_file
    loss_fn = args.loss_fn
    eval_file = args.eval_file
    lr = args.lr

    args = vars(args)

    if args['model_type'] == 'bert':
        args['bert_model'] = 'bert-base-uncased'
    elif args['model_type'] == 'roberta':
        args['bert_model'] = 'roberta-base'
    elif args['model_type'] == 'transformer':
        if args['bert_model'] is None:
            raise ValueError("Need to specify a bert_model for model_type transformer!")
    else:
        raise ValueError("Unknown model type " + args['model_type'])

    if os.path.exists(save_name) and not args.get('force', False):
        raise FileExistsError(f"Save name {save_name} already exists. Training would override existing data. Aborting...")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file {train_file} not found. Try again with a valid path.")

    logger.info("Running training script with the following args:")
    for arg in args:
        logger.info(f"{arg}: {args[arg]}")
    logger.info("------------------------------------------------------------")

    trainer = TransformerBaselineTrainer(model_args=args, transformer_name=args['bert_model'], loss_func=loss_fn, lr=lr)

    trainer.train(num_epochs=num_epochs, save_name=save_name, train_file=train_file, args=args, eval_file=eval_file)
    return trainer

if __name__ == "__main__":
    main()
