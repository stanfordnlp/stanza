import torch
import torch.nn as nn
import os
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from typing import List, Tuple

import utils 
from constants import * 
# from stanza.models.lemma_classifier import utils
# from stanza.models.lemma_classifier.constants import *

class LemmaClassifier(nn.Module):
    """
    Model architecture:
        Extracts word embeddings over the sentence, passes embeddings into a bi-LSTM to get a sentence encoding.
        From the LSTM output, we get the embedding fo the specific token that we classify on. That embedding 
        is fed into an MLP for classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embeddings, padding_idx = 0, **kwargs):
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

        # Embedding layer with GloVe embeddings
        self.glove = get_glove(self.embedding_dim)
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)

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

    def forward(self, token_ids: torch.tensor, pos_index: int, words: List[str]):
        """
        Computes the forward pass of the neural net

        Args:
            token_ids (torch.tensor): Tensor of the tokenized indices of the words in the input sentence, with unknown words having their index set to UNKNOWN_TOKEN_IDX
            pos_index (int): The position index of the target token for lemmatization classification in the sentence.
            words (List[str]): A list of the tokenized strings of the input sentence.

        Returns:
            torch.tensor: Output logits of the neural network
        """
        
        # UNKNOWN_TOKEN will be our <UNK> token
        # UNKNOWN_TOKEN_IDX will be the custom index for the <UNK> token
        unk_token_indices = utils.extract_unknown_token_indices(token_ids, UNKNOWN_TOKEN_IDX)
        unknown_mask = (token_ids == UNKNOWN_TOKEN_IDX)
        masked_indices = token_ids.masked_fill(unknown_mask, 0)  # Replace UNKNOWN_TOKEN_IDX with 0 for embedding lookup

        # replace 0 token vectors with the true unknown 
        embedded = self.embedding_layer(masked_indices)
        for unk_token_idx in unk_token_indices:
            embedded[unk_token_idx] = self.glove[UNKNOWN_TOKEN]
        
        if self.use_charlm:
            char_reps_forward = self.charmodel_forward.build_char_representation([words])  # takes [[str]]
            char_reps_backward = self.charmodel_backward.build_char_representation([words])
        
            embedded = torch.cat((embedded, char_reps_forward[0], char_reps_backward[0]), 1)   # take [0] because we only use the first sentence
        
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Extract the hidden state at the index of the token to classify
        lstm_out = lstm_out[pos_index]

        # MLP forward pass
        output = self.mlp(lstm_out)
        return output
