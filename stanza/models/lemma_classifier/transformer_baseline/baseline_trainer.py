"""
This file contains code used to train a baseline transformer model to classify on a lemma of a particular token. 
"""

from model import LemmaClassifierWithTransformer
import torch.nn as nn 
import torch 
import torch.optim as optim
from typing import List, Tuple, Any 
from os import path
import os
import sys

parent_dir = os.path.dirname(__file__)
above_dir = os.path.dirname(parent_dir)
sys.path.append(above_dir)

import utils

class TransformerBaselineTrainer:
    """
    Class to assist with training a baseline transformer model to classify on token lemma
    """

    def __init__(self, output_dim: int):
        self.output_dim = output_dim 

        self.model = LemmaClassifierWithTransformer(output_dim=self.output_dim)
        self.criterion = nn.CrossEntropyLoss()  # TODO maybe make this custom
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # TODO maybe also make this custom


    def train(self, texts_batch: List[List[str]], positions_batch: List[int], labels_batch: List[int], num_epochs: int, save_name: str, **kwargs):

        if kwargs.get("from_file"):
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
            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


        torch.save(self.model.state_dict(), save_name)
        print(f"Saved model state dict to {save_name}")


def main():

    # Train on single set of examples
    demo_model_path = path.join(path.dirname(__file__), "demo_model.pt")
    if os.path.exists(demo_model_path):
        os.remove(demo_model_path)
    
    trainer = TransformerBaselineTrainer(output_dim=64)
    
    tokenized_sentence = ['the', 'cat', "'s", 'tail', 'is', 'long']
    text_batches = [tokenized_sentence]
    # Convert the tokenized input to a tensor
    
    positional_index = tokenized_sentence.index("'s")
    target = torch.tensor(0, dtype=torch.long)  # 0 for "be" and 1 for "have"
    index_batches = [positional_index]
    target_batches = [target]
    # Train
    trainer.train(text_batches, index_batches, target_batches, 10, path.join(path.dirname(__file__), demo_model_path))

    train_file = path.join(path.dirname(path.dirname(__file__)), "test_output.txt")
    trainer.train([], [], [], 10, "big_demo_model.pt", from_file=True, train_path=train_file, label_decoder={"be": 0, "have": 1})


if __name__ == "__main__":
    main()
