"""
This file contains code used to train a baseline transformer model to classify on a lemma of a particular token. 
"""

import torch.nn as nn 
import torch 
import torch.optim as optim
from typing import List, Tuple, Any 
import argparse
import os
import sys
import logging

from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.constants import ModelType
from stanza.models.lemma_classifier.evaluate_models import evaluate_model
from stanza.models.lemma_classifier.transformer_baseline.model import LemmaClassifierWithTransformer
from stanza.utils.get_tqdm import get_tqdm
from stanza.models.common.utils import default_device
from typing import Mapping, List, Tuple, Any

tqdm = get_tqdm()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TransformerBaselineTrainer:
    """
    Class to assist with training a baseline transformer model to classify on token lemmas.
    To find the model spec, refer to `model.py` in this directory.
    """

    def __init__(self, output_dim: int = 2, transformer_name: str = "roberta", loss_func: str = "ce", lr: int = 0.001):
        """
        Creates the Trainer object

        Args:
            output_dim (int, optional): The dimension of the output layer from the MLP in the classifier model. Defaults to 2.
            transformer_name (str, optional): What kind of transformer to use for embeddings. Defaults to "roberta".
            loss_func (str, optional): Which loss function to use (either 'ce' or 'weighted_bce'). Defaults to "ce".
            lr (int, optional): learning rate for the optimizer. Defaults to 0.001.
        """
        self.output_dim = output_dim 

        self.model = LemmaClassifierWithTransformer(output_dim=self.output_dim, transformer_name=transformer_name)
        # Find loss function
        if loss_func == "ce":
            self.criterion = nn.CrossEntropyLoss()
            self.weighted_loss = False
        elif loss_func == "weighted_bce":
            self.criterion = nn.BCEWithLogitsLoss()  
            self.weighted_loss = True  # used to add weights during train time.
        else:
            raise ValueError("Must enter a valid loss function (e.g. 'ce' or 'weighted_bce')")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) 

    def configure_weighted_loss(self, label_decoder: Mapping, counts: Mapping):
        """
        If applicable, this function will update the loss function of the LemmaClassifier model to become BCEWithLogitsLoss.
        The weights are determined by the counts of the classes in the dataset. The weights are inversely proportional to the
        frequency of the class in the set. E.g. classes with lower frequency will have higher weight.
        """
        weights = [0 for _ in label_decoder.keys()]  # each key in the label decoder is one class, we have one weight per class
        total_samples = sum(counts.values())
        for class_idx in counts:
            weights[class_idx] = total_samples / (counts[class_idx] * len(counts))  # weight_i = total / (# examples in class i * num classes)
        weights = torch.tensor(weights)
        logging.info(f"Using weights {weights} for weighted loss.")
        self.criterion = nn.BCEWithLogitsLoss(weight=weights)

    def save_checkpoint(self, save_name: str, state_dict: Mapping, label_decoder: Mapping, args: Mapping) -> Mapping:
        """
        Saves model checkpoint with a current state dict (params) and a label decoder on the dataset.
        If the save path doesn't exist, it will create it. 
        """
        save_dir = os.path.split(save_name)[0]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        state_dict = {
            "params": state_dict,
            "label_decoder": label_decoder,
            "model_type": ModelType.TRANSFORMER,
            "args": args,
        }
        torch.save(state_dict, save_name)
        return state_dict

    def train(self, num_epochs: int, save_name: str, args: Mapping, eval_file: str, **kwargs):

        """
        Trains a model on batches of texts, position indices of the target token, and labels (lemma annotation) for the target token.

        Args:
            texts_batch (List[List[str]]): Batches of tokenized texts, one per sentence. Expected to contain at least one instance of the target token.
            positions_batch (List[int]): Batches of position indices (zero-indexed) for the target token, one per input sentence. 
            labels_batch (List[int]): Batches of labels for the target token, one per input sentence. 
            num_epochs (int): Number of training epochs
            save_name (str): Path to file where trained model should be saved. 
        
        Kwargs:
            train_path (str): Path to data file, containing tokenized text sentences, token index and true label for token lemma on each line. 
            eval_file (str): Path to the dev set file for evaluating model checkpoints each epoch.
        """

        # Put model on GPU (if possible)  
        device = default_device()
        self.model.to(device)
        self.model.transformer.to(device)


        if kwargs.get("train_path"):
            texts_batch, positions_batch, labels_batch, counts, label_decoder = utils.load_dataset(kwargs.get("train_path"), get_counts=self.weighted_loss)
            self.output_dim = len(label_decoder)
            logging.info(f"Using label decoder : {label_decoder}")

            # Move data to device
            labels_batch = torch.tensor(labels_batch, device=device)
            positions_batch = torch.tensor(positions_batch, device=device)
        
        assert len(texts_batch) == len(positions_batch) == len(labels_batch), f"Input batch sizes did not match ({len(texts_batch)}, {len(positions_batch)}, {len(labels_batch)})."
        if os.path.exists(save_name):
            raise FileExistsError(f"Save name {save_name} already exists; training would overwrite previous file contents. Aborting...")
        
        if self.weighted_loss:
            self.configure_weighted_loss(label_decoder, counts)

        selected_dev = next(self.model.transformer.parameters()).device
        self.criterion = self.criterion.to(selected_dev)

        best_model, best_f1 = None, float("-inf")
        for epoch in range(num_epochs):
            # go over entire dataset with each epoch
            for texts, position, label in tqdm(zip(texts_batch, positions_batch, labels_batch), total=len(texts_batch)):
                if position < 0 or position > len(texts) - 1:  # validate position index
                    raise ValueError(f"Found position {position} in text: {texts}, which is not possible.") 
                
                self.optimizer.zero_grad()
                output = self.model(position, texts)
                
                # Compute loss, which is different if using CE or BCEWithLogitsLoss
                if self.weighted_loss:  # BCEWithLogitsLoss requires a vector for target where probability is 1 on the true label class, and 0 on others.
                    target_vec = [1, 0] if label == 0 else [0, 1]
                    target = torch.tensor(target_vec, dtype=torch.float32, device=device)
                else:  # CELoss accepts target as just raw label
                    target = torch.tensor(label, dtype=torch.long, device=device)
                
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
            
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
            if eval_file:
                # Evaluate model on dev set to see if it should be saved.
                _, _, _, f1 = evaluate_model(self.model, label_decoder, eval_file, is_training=True)
                logging.info(f"Weighted f1 for model: {f1}")
                if f1 > best_f1:
                    best_f1 = f1
                    self.save_checkpoint(save_name, self.model.state_dict(), label_decoder, args)
                    logging.info(f"New best model: weighted f1 score of {f1}.")
            else:
                self.save_checkpoint(save_name, self.model.state_dict(), label_decoder, args)


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dim", type=int, default=2, help="Size of output layer (number of classes)")
    parser.add_argument("--save_name", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "big_model_roberta_weighted_loss.pt"), help="Path to model save file")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_sets", "combined_train.txt"), help="Full path to training file")
    parser.add_argument("--model_type", type=str, default="roberta", help="Which transformer to use ('bert' or 'roberta')")
    parser.add_argument("--bert_model", type=str, default=None, help="Use a specific transformer instead of the default bert/roberta")
    parser.add_argument("--loss_fn", type=str, default="weighted_bce", help="Which loss function to train with (e.g. 'ce' or 'weighted_bce')")
    parser.add_argument("--eval_file", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_sets", "combined_dev.txt"), help="Path to dev file used to evaluate model for saves")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")

    args = parser.parse_args(args)

    output_dim = args.output_dim
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

    if os.path.exists(save_name):
        raise FileExistsError(f"Save name {save_name} already exists. Training would override existing data. Aborting...")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file {train_file} not found. Try again with a valid path.")

    logging.info("Running training script with the following args:")
    for arg in args:
        logging.info(f"{arg}: {args[arg]}")
    logging.info("------------------------------------------------------------")
    
    trainer = TransformerBaselineTrainer(output_dim=output_dim, transformer_name=args['bert_model'], loss_func=loss_fn, lr=lr)

    trainer.train(num_epochs=num_epochs, save_name=save_name, train_path=train_file, args=args, eval_file=eval_file)


if __name__ == "__main__":
    main()
