import torch
import torch.nn as nn
import os
import sys
import logging

from transformers import AutoTokenizer, AutoModel
from typing import Mapping, List, Tuple, Any
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.constants import ModelType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LemmaClassifierWithTransformer(LemmaClassifier):
    def __init__(self, output_dim: int, transformer_name: str, label_decoder: Mapping):
        """
        Model architecture:

            Use a transformer (BERT or RoBERTa) to extract contextual embedding over a sentence.
            Get the embedding for the word that is to be classified on, and feed the embedding
            as input to an MLP classifier that has 2 linear layers, and a prediction head.

        Args:
            output_dim (int): Dimension of the output from the MLP 
            model_type (str): What kind of transformer to use for embeddings ('bert' or 'roberta')
        """
        super(LemmaClassifierWithTransformer, self).__init__()

        # Choose transformer
        self.transformer_name = transformer_name
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True, add_prefix_space=True)
        self.transformer = AutoModel.from_pretrained(transformer_name)
        config = self.transformer.config

        embedding_size = config.hidden_size

        # define an MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.label_decoder = label_decoder

    def forward(self, idx_positions: List[int], sentences: List[List[str]]):
        """

        Args:
            text (List[str]): A single sentence with each token as an entry in the list.
            pos_index (int): The index of the token to classify on.

        Returns the logits of the MLP
        """

        # Get the transformer embedding IDs for each token in each sentence
        device = next(self.transformer.parameters()).device
        tokenized_inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", is_split_into_words=True)
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
        
        # Forward pass through Transformer
        outputs = self.transformer(**tokenized_inputs)
        
        # Get embeddings for all tokens
        last_hidden_state = outputs.last_hidden_state

        embeddings = last_hidden_state[torch.arange(last_hidden_state.size(0)), idx_positions]
        # pass to the MLP
        output = self.mlp(embeddings)
        return output

    def model_type(self):
        return ModelType.TRANSFORMER
