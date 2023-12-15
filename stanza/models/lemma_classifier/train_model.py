"""
The code in this file works to train a lemma classifier for 's
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import utils
import os 
import logging
import argparse
from os import path
from os import remove
from model import LemmaClassifier
from typing import List, Tuple, Any
from constants import get_glove, UNKNOWN_TOKEN_IDX


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LemmaClassifierTrainer():
    """
    Class to assist with training a LemmaClassifier
    """

    def __init__(self, vocab_size: int, embeddings: str, embedding_dim: int, hidden_dim: int, output_dim: int, use_charlm: bool, **kwargs):
        """
        Initializes the LemmaClassifierTrainer class.
        
        Args:
            vocab_size (int): Size of the vocab being used (if custom vocab)
            embeddings (str): What word embeddings to use (currently only supports GloVe) TODO add more!
            embedding_dim (int): Size of embedding dimension to use on the aforementioned word embeddings
            hidden_dim (int): Size of hidden vectors in LSTM layers
            output_dim (int): Size of output vector from MLP layer
            use_charlm (bool): Whether to use charlm embeddings as well

        Kwargs:
            forward_charlm_file (str): Path to the forward pass embeddings for the charlm 
            backward_charlm_file (str): Path to the backward pass embeddings for the charlm
            lr (float): Learning rate, defaults to 0.001.

        Raises:
            FileNotFoundError: If the forward charlm file is not present
            FileNotFoundError: If the backward charlm file is not present
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Load word embeddings
        self.embeddings = None
        if embeddings == "glove":
            self.embeddings = get_glove(embedding_dim)
            self.vocab_size = len(self.embeddings.itos)

        # Load CharLM embeddings
        forward_charlm_file = kwargs.get("forward_charlm_file")
        backward_charlm_file = kwargs.get("backward_charlm_file")
        if use_charlm and forward_charlm_file is not None and not os.path.exists(forward_charlm_file):
            raise FileNotFoundError(f"Could not find foward charlm file: {forward_charlm_file}")
        if use_charlm and backward_charlm_file is not None and not os.path.exists(backward_charlm_file):
            raise FileNotFoundError(f"Could not find foward charlm file: {backward_charlm_file}")

        self.model = LemmaClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, self.embeddings.vectors, charlm=use_charlm,
                                     charlm_forward_file=forward_charlm_file, charlm_backward_file=backward_charlm_file)
        self.criterion = nn.CrossEntropyLoss()  
        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs.get("lr", 0.001))  

    def train(self, texts_batch: List[List[str]], positions_batch: List[int], labels_batch: List[int], num_epochs: int, save_name: str, **kwargs) -> None:

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

        train_path, label_decoder = kwargs.get("train_path"), kwargs.get("label_decoder", {})
        if train_path:  # use file to train model
            texts_batch, positions_batch, labels_batch = utils.load_dataset(train_path, label_decoder=label_decoder)
            logging.info(f"Loaded dataset successfully from {train_path}")
            logging.info(f"Using label decoder: {label_decoder}")

        assert len(texts_batch) == len(positions_batch) == len(labels_batch), f"Input batch sizes did not match ({len(texts_batch)}, {len(positions_batch)}, {len(labels_batch)})."
        if path.exists(save_name):
            raise FileExistsError(f"Save name {save_name} already exists; training would overwrite previous file contents. Aborting...")
    
        for epoch in range(num_epochs):
            # go over entire dataset with each epoch
            for texts, position, label in zip(texts_batch, positions_batch, labels_batch):
                if position < 0 or position > len(texts) - 1:  # validate position index
                    raise ValueError(f"Found position {position} in text: {texts}, which is not possible.")
                
                # Any token not in self.embeddings.stoi will be given the UNKNOWN_TOKEN_IDX, which is resolved to a true embedding in LemmaClassifier's forward() func
                token_ids = torch.tensor([self.embeddings.stoi[word.lower()] if word.lower() in self.embeddings.stoi else UNKNOWN_TOKEN_IDX for word in texts])  
                
                self.optimizer.zero_grad()

                output = self.model(token_ids, position, texts)
                target = torch.tensor(label, dtype=torch.long)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
            
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


        torch.save(self.model.state_dict(), save_name)
        logging.info(f"Saved model state dict to {save_name}")


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000, help="Number of tokens in vocab")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Number of dimensions in word embeddings (currently using GloVe)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Size of hidden layer")
    parser.add_argument("--output_dim", type=int, default=2, help="Size of output layer (number of classes)")
    parser.add_argument("--use_charlm", type=bool, default=True, help="Whether not to use the charlm embeddings")
    parser.add_argument("--forward_charlm_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_forward.pt"), help="Path to forward charlm file")
    parser.add_argument("--backward_charlm_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_backwards.pt"), help="Path to backward charlm file")
    parser.add_argument("--save_name", type=str, default=path.join(path.dirname(__file__), "saved_models", "lemma_classifier_model.pt"), help="Path to model save file")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(__file__), "test_output.txt"), help="Full path to training file")

    args = parser.parse_args()

    vocab_size = args.vocab_size
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    use_charlm = args.use_charlm
    forward_charlm_file = args.forward_charlm_file
    backward_charlm_file = args.backward_charlm_file
    save_name = args.save_name 
    lr = args.lr
    num_epochs = args.num_epochs
    train_file = args.train_file

    if os.path.exists(save_name):
        raise FileExistsError(f"Save name {save_name} already exists. Training would override existing data. Aborting...")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file {train_file} not found. Try again with a valid path.")

    trainer = LemmaClassifierTrainer(vocab_size=vocab_size,
                                     embeddings="glove",
                                     embedding_dim=embedding_dim,
                                     hidden_dim=hidden_dim,
                                     output_dim=output_dim,
                                     use_charlm=use_charlm,
                                     forward_charlm_file=forward_charlm_file,
                                     backward_charlm_file=backward_charlm_file,
                                     lr=lr
                                     )
    trainer.train(
        [], [], [], num_epochs=num_epochs, save_name=save_name, train_path=train_file, label_decoder={"be": 0, "have": 1}
    )
