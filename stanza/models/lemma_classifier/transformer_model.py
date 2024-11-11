import torch
import torch.nn as nn
import os
import sys
import logging

from transformers import AutoTokenizer, AutoModel
from typing import Mapping, List, Tuple, Any
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from stanza.models.common.bert_embedding import extract_bert_embeddings
from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.constants import ModelType

logger = logging.getLogger('stanza.lemmaclassifier')

class LemmaClassifierWithTransformer(LemmaClassifier):
    def __init__(self, model_args: dict, output_dim: int, transformer_name: str, label_decoder: Mapping, target_words: set, target_upos: set):
        """
        Model architecture:

            Use a transformer (BERT or RoBERTa) to extract contextual embedding over a sentence.
            Get the embedding for the word that is to be classified on, and feed the embedding
            as input to an MLP classifier that has 2 linear layers, and a prediction head.

        Args:
            model_args (dict): args for the model
            output_dim (int): Dimension of the output from the MLP
            transformer_name (str): name of the HF transformer to use
            label_decoder (dict): a map of the labels available to the model
            target_words (set(str)): a set of the words which might need lemmatization
        """
        super(LemmaClassifierWithTransformer, self).__init__(label_decoder, target_words, target_upos)
        self.model_args = model_args

        # Choose transformer
        self.transformer_name = transformer_name
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True, add_prefix_space=True)
        self.add_unsaved_module("transformer", AutoModel.from_pretrained(transformer_name))
        config = self.transformer.config

        embedding_size = config.hidden_size

        # define an MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def get_save_dict(self):
        save_dict = {
            "params": self.state_dict(),
            "label_decoder": self.label_decoder,
            "target_words": list(self.target_words),
            "target_upos": list(self.target_upos),
            "model_type": self.model_type().name,
            "args": self.model_args,
        }
        skipped = [k for k in save_dict["params"].keys() if self.is_unsaved_module(k)]
        for k in skipped:
            del save_dict["params"][k]
        return save_dict

    def convert_tags(self, upos_tags: List[List[str]]):
        return None

    def forward(self, idx_positions: List[int], sentences: List[List[str]], upos_tags: List[List[int]]):
        """
        Computes the forward pass of the transformer baselines

        Args:
            idx_positions (List[int]): A list of the position index of the target token for lemmatization classification in each sentence.
            sentences (List[List[str]]): A list of the token-split sentences of the input data.
            upos_tags (List[List[int]]): A list of the upos tags for each token in every sentence - not used in this model, here for compatibility

        Returns:
            torch.tensor: Output logits of the neural network, where the shape is  (n, output_size) where n is the number of sentences.
        """
        device = next(self.transformer.parameters()).device
        bert_embeddings = extract_bert_embeddings(self.transformer_name, self.tokenizer, self.transformer, sentences, device,
                                                  keep_endpoints=False, num_layers=1, detach=True)
        embeddings = [emb[idx] for idx, emb in zip(idx_positions, bert_embeddings)]
        embeddings = torch.stack(embeddings, dim=0)[:, :, 0]
        # pass to the MLP
        output = self.mlp(embeddings)
        return output

    def model_type(self):
        return ModelType.TRANSFORMER
