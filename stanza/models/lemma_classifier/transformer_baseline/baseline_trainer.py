"""
This file contains code used to train a baseline transformer model to classify on a lemma of a particular token. 
"""

from model import LemmaClassifierWithTransformer
import torch.nn as nn 
import torch 
import torch.optim as optim
from typing import List, Tuple, Any 
from os import path
import argparse
import os
import sys
import logging

parent_dir = os.path.dirname(__file__)
above_dir = os.path.dirname(parent_dir)
sys.path.append(above_dir)

import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TransformerBaselineTrainer:
    """
    Class to assist with training a baseline transformer model to classify on token lemmas.
    To find the model spec, refer to `model.py` in this directory.
    """

    def __init__(self, output_dim: int, model_type: str):
        """
        Creates the Trainer object

        Args:
            output_dim (int): The dimension of the output layer from the MLP in the classifier model.
            model_type (str): What kind of transformer to use for embeddings ('bert' or 'roberta')
        """
        self.output_dim = output_dim 

        self.model = LemmaClassifierWithTransformer(output_dim=self.output_dim, model_type=model_type)
        self.criterion = nn.CrossEntropyLoss() 
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
            label_decoder (Mapping[str, int]): A map between target token lemmas and their corresponding integers for the labels 

        """

        if kwargs.get("train_path"):
            texts_batch, positions_batch, labels_batch = utils.load_dataset(kwargs.get("train_path"), label_decoder=kwargs.get("label_decoder"))
        
        assert len(texts_batch) == len(positions_batch) == len(labels_batch), f"Input batch sizes did not match ({len(texts_batch)}, {len(positions_batch)}, {len(labels_batch)})."
        if path.exists(save_name):
            raise FileExistsError(f"Save name {save_name} already exists; training would overwrite previous file contents. Aborting...")

        for epoch in range(num_epochs):
            # go over entire dataset with each epoch
            for texts, position, label in zip(texts_batch, positions_batch, labels_batch):
                if position < 0 or position > len(texts) - 1:  # validate position index
                    raise ValueError(f"Found position {position} in text: {texts}, which is not possible.") 
                
                self.optimizer.zero_grad()
                output = self.model(texts, position)
                target = torch.tensor(label, dtype=torch.long)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
            
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


        torch.save(self.model.state_dict(), save_name)
        logging.info(f"Saved model state dict to {save_name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dim", type=int, default=2, help="Size of output layer (number of classes)")
    parser.add_argument("--save_name", type=str, default=path.join(path.dirname(path.dirname(__file__)), "saved_models", "big_model_roberta.pt"), help="Path to model save file")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_output.txt"), help="Full path to training file")
    parser.add_argument("--model_type", type=str, default="roberta", help="Which transformer to use ('bert' or 'roberta')")

    args = parser.parse_args()

    output_dim = args.output_dim
    save_name = args.save_name
    num_epochs = args.num_epochs
    train_file = args.train_file
    model_type = args.model_type

    if os.path.exists(save_name):
        raise FileExistsError(f"Save name {save_name} already exists. Training would override existing data. Aborting...")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file {train_file} not found. Try again with a valid path.")
    
    trainer = TransformerBaselineTrainer(output_dim=output_dim, model_type=model_type)

    trainer.train([], [], [], num_epochs=num_epochs, save_name=save_name, train_path=train_file, label_decoder={"be": 0, "have": 1})


if __name__ == "__main__":
    main()
