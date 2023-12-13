"""
The code in this file works to train a lemma classifier for 's
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import utils
import os 
from os import path
from os import remove
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer
from model import LemmaClassifier
from typing import List, Tuple, Any
from constants import get_glove, UNKNOWN_TOKEN_IDX


class LemmaClassifierTrainer():
    """
    Class to assist with training a LemmaClassifier
    """

    def __init__(self, vocab_size: int, embeddings: str, embedding_dim: int, hidden_dim: int, output_dim: int, use_charlm: bool, **kwargs):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embeddings = None
        if embeddings == "glove":
            self.embeddings = get_glove(embedding_dim)
            self.vocab_size = len(self.embeddings.itos)

        self.model = LemmaClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, self.embeddings.vectors, charlm=use_charlm,
                                     charlm_forward_file=kwargs.get("forward_charlm_file"), charlm_backward_file=kwargs.get("backward_charlm_file"))
        self.criterion = nn.CrossEntropyLoss()  
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  

    def train(self, texts_batch: List[List[str]], positions_batch: List[int], labels_batch: List[int], num_epochs: int, save_name: str, **kwargs) -> None:

        """
        Trains a model on batches of texts, position indices of the target token, and labels (lemma annotation) for the target token.

        Args:
            texts_batch (List[List[str]]): Batches of tokenized texts, one per sentence. Expected to contain at least one instance of the target token.
            positions_batch (List[int]): Batches of position indices (zero-indexed) for the target token, one per input sentence. 
            labels_batch (List[int]): Batches of labels for the target token, one per input sentence. 
            num_epochs (int): Number of training epochs
            save_name (str): Path to file where trained model should be saved. 
        """

        if kwargs.get("train_path"):
            texts_batch, positions_batch, labels_batch = utils.load_dataset(kwargs.get("train_path"), label_decoder=kwargs.get("label_decoder", {}))
            print(f"Loaded dataset successfully from {kwargs.get('train_path')}")
            print(f"Label decoder: {kwargs.get('label_decoder')}")

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
            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


        torch.save(self.model.state_dict(), save_name)
        print(f"Saved model state dict to {save_name}")


if __name__ == "__main__":

    demo_model_path = path.join(path.dirname(__file__), "demo_model.pt")
    if path.exists(demo_model_path):
        remove(demo_model_path)

    # Define the hyperparameters
    vocab_size = 10000  # Adjust based on your dataset
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 2  # Binary classification (be or have)

    trainer = LemmaClassifierTrainer(vocab_size=vocab_size, 
                                     embeddings="glove",
                                     embedding_dim=embedding_dim,
                                     hidden_dim=hidden_dim,
                                     output_dim=output_dim,
                                     use_charlm=True,
                                     forward_charlm_file=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_forward.pt"),
                                     backward_charlm_file=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_backwards.pt"))
    
    tokenized_sentence = ['the', 'cat', "'s", 'tail', 'is', 'long']
    text_batches = [tokenized_sentence]

    # Convert the tokenized input to a tensor
    positional_index = tokenized_sentence.index("'s")
    target = torch.tensor(0, dtype=torch.long)  # 0 for "be" and 1 for "have"
    index_batches = [positional_index]
    target_batches = [target]
    # Train
    trainer.train(text_batches, index_batches, target_batches, 10, path.join(path.dirname(__file__), demo_model_path))

    train_file = path.join(path.dirname(__file__), "test_output.txt")
    trainer.train([], [], [], 10, "big_demo_model.pt", train_path=train_file, label_decoder={"be": 0, "have": 1})
