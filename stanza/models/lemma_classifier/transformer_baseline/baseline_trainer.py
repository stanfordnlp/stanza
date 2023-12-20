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
from stanza.models.lemma_classifier.transformer_baseline.model import LemmaClassifierWithTransformer
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TransformerBaselineTrainer:
    """
    Class to assist with training a baseline transformer model to classify on token lemmas.
    To find the model spec, refer to `model.py` in this directory.
    """

    def __init__(self, output_dim: int, transformer_name: str, loss_func: str):
        """
        Creates the Trainer object

        Args:
            output_dim (int): The dimension of the output layer from the MLP in the classifier model.
            transformer_name (str): What kind of transformer to use for embeddings
            loss_func (str): Which loss function to use (either 'ce' or 'weighted_bce') 
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
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) 

    def train(self, texts_batch: List[List[str]], positions_batch: List[int], labels_batch: List[int], num_epochs: int, save_name: str, **kwargs):

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
        """

        if kwargs.get("train_path"):
            texts_batch, positions_batch, labels_batch, counts, label_decoder = utils.load_dataset(kwargs.get("train_path"), get_counts=self.weighted_loss)
            logging.info(f"Using label decoder : {label_decoder}")
        
        assert len(texts_batch) == len(positions_batch) == len(labels_batch), f"Input batch sizes did not match ({len(texts_batch)}, {len(positions_batch)}, {len(labels_batch)})."
        if os.path.exists(save_name):
            raise FileExistsError(f"Save name {save_name} already exists; training would overwrite previous file contents. Aborting...")
        
        # Configure weighted loss, if necessary
        if self.weighted_loss:
            weights = [0 for _ in label_decoder.keys()]  # each key in the label decoder is one class, we have one weight per class
            total_samples = sum(counts.values())
            for class_idx in counts:
                weights[class_idx] = total_samples / (counts[class_idx] * len(counts))  # weight_i = total / (# examples in class i * num classes)
                weights = torch.tensor(weights)
                logging.info(f"WEIGHTS, {weights}")
            self.criterion = nn.BCEWithLogitsLoss(weight=weights)

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
                    target = torch.tensor(target_vec, dtype=torch.float32)
                else:  # CELoss accepts target as just raw label
                    target = torch.tensor(label, dtype=torch.long)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
            
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


        save_dir = os.path.split(save_name)[0]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        state_dict = {
            "params": self.model.state_dict(),
            "label_decoder": label_decoder,
        }
        torch.save(state_dict, save_name)
        logging.info(f"Saved model state dict to {save_name}")


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dim", type=int, default=2, help="Size of output layer (number of classes)")
    parser.add_argument("--save_name", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "big_model_bert_weighted_loss.pt"), help="Path to model save file")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_sets", "combined_train.txt"), help="Full path to training file")
    parser.add_argument("--model_type", type=str, default="roberta", help="Which transformer to use ('bert' or 'roberta')")
    parser.add_argument("--bert_model", type=str, default=None, help="Use a specific transformer instead of the default bert/roberta")
    parser.add_argument("--loss_fn", type=str, default="weighted_bce", help="Which loss function to train with (e.g. 'ce' or 'weighted_bce')")

    args = parser.parse_args(args)

    output_dim = args.output_dim
    save_name = args.save_name
    num_epochs = args.num_epochs
    train_file = args.train_file
    loss_fn = args.loss_fn

    if args.bert_model is None:
        if args.model_type == 'bert':
            transformer_name = 'bert-base-uncased'
        elif args.model_type == 'roberta':
            transformer_name = 'roberta-base'
        else:
            raise ValueError("Unknown model type " + args.model_type)
    else:
        transformer_name = args.bert_model

    if os.path.exists(save_name):
        raise FileExistsError(f"Save name {save_name} already exists. Training would override existing data. Aborting...")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file {train_file} not found. Try again with a valid path.")

    logging.info("Running training script with the following args:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("------------------------------------------------------------")
    
    trainer = TransformerBaselineTrainer(output_dim=output_dim, transformer_name=transformer_name, loss_func=loss_fn)

    trainer.train([], [], [], num_epochs=num_epochs, save_name=save_name, train_path=train_file)


if __name__ == "__main__":
    main()
