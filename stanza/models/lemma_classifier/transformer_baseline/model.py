import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer
from transformers import BertTokenizer, BertModel
from typing import List, Tuple, Any 


class LemmaClassifierWithTransformer(nn.Module):

    def __init__(self, output_dim):
        super(LemmaClassifierWithTransformer, self).__init__()

        # Get the embedding through transformer 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        config = self.bert.config 
        embedding_size = config.hidden_size

        # define an MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, text: List[str], pos_index: int):

        # Get the transformer embeddings 
        input_ids = self.tokenizer.convert_tokens_to_ids(text)

        # Convert tokens to IDs and put them into a tensor
        input_ids_tensor = torch.tensor([input_ids])

        # Forward pass through BERT
        with torch.no_grad():
            outputs = self.bert(input_ids_tensor)
        
        # Get embeddings for all tokens
        last_hidden_state = outputs.last_hidden_state
        token_embeddings = last_hidden_state[0]

        # Get target embedding
        target_pos_embedding = token_embeddings[pos_index]

        # pass to the MLP
        output = self.mlp(target_pos_embedding)
        return output
