import torch
import torch.nn as nn
import os
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from typing import List, Tuple

from stanza.models.common.vocab import UNK_ID
from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.constants import *

class LemmaClassifier(nn.Module):
    """
    Model architecture:
        Extracts word embeddings over the sentence, passes embeddings into a bi-LSTM to get a sentence encoding.
        From the LSTM output, we get the embedding fo the specific token that we classify on. That embedding 
        is fed into an MLP for classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab_map, pt_embedding, padding_idx = 0, **kwargs):
        """
        Args:
            vocab_size (int): Size of the vocab being used (if custom vocab)
            embeddings (word vectors for embedding): What word embeddings to use (currently only supports GloVe) TODO add more!
            embedding_dim (int): Size of embedding dimension to use on the aforementioned word embeddings
            hidden_dim (int): Size of hidden vectors in LSTM layers
            output_dim (int): Size of output vector from MLP layer
            padding_idx (int, optional): Padding index for the embedding layerr. Defaults to 0.

        Kwargs:
            charlm (bool): Whether or not to use the charlm embeddings
            charlm_forward_file (str): The path to the forward pass model for the character language model
            charlm_backward_file (str): The path to the forward pass model for the character language model.
        
        Raises:
            FileNotFoundError: if the forward or backward charlm file cannot be found.
        """
        super(LemmaClassifier, self).__init__()
        self.input_size = 0
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)
    
        self.embedding_dim = embedding_dim
        self.input_size += embedding_dim

        self.embedding = pt_embedding
        self.vocab_map = vocab_map

        # Optionally, include charlm embeddings  
        self.use_charlm = kwargs.get("charlm")

        if self.use_charlm:
            charlm_forward_file = kwargs.get("charlm_forward_file")
            charlm_backward_file = kwargs.get("charlm_backward_file")
            if charlm_forward_file is None or not os.path.exists(charlm_forward_file):
                raise FileNotFoundError(f'Could not find forward character model: {kwargs.get("charlm_forward_file", "FILE_NOT_PROVIDED")}')
            if charlm_backward_file is None or not os.path.exists(charlm_backward_file):
                raise FileNotFoundError(f'Could not find backward character model: {kwargs.get("charlm_backward_file", "FILE_NOT_PROVIDED")}')
            add_unsaved_module('charmodel_forward', CharacterLanguageModel.load(charlm_forward_file, finetune=False))
            add_unsaved_module('charmodel_backward', CharacterLanguageModel.load(charlm_backward_file, finetune=False))
            
            self.input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
        
        self.lstm = nn.LSTM(
            self.input_size, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
                        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, pos_index: int, words: List[str]):
        """
        Computes the forward pass of the neural net

        Args:
            pos_index (int): The position index of the target token for lemmatization classification in the sentence.
            words (List[str]): A list of the tokenized strings of the input sentence.

        Returns:
            torch.tensor: Output logits of the neural network
        """
        token_ids = [self.vocab_map.get(word.lower(), UNK_ID) for word in words]
        token_ids = torch.tensor(token_ids, device=self.device)
        embedded = self.embedding(token_ids)

        if self.use_charlm:
            char_reps_forward = self.charmodel_forward.build_char_representation([words])  # takes [[str]]
            char_reps_backward = self.charmodel_backward.build_char_representation([words])

            embedded = torch.cat((embedded, char_reps_forward[0], char_reps_backward[0]), 1)   # take [0] because we only use the first sentence

        lstm_out, (hidden, _) = self.lstm(embedded)

        # Extract the hidden state at the index of the token to classify
        pos_index = torch.tensor(pos_index, device=self.device)
        lstm_out = lstm_out[pos_index]

        # MLP forward pass
        output = self.mlp(lstm_out)
        return output
