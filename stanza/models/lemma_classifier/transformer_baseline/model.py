import torch
import torch.nn as nn
import os
import sys
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
from typing import List, Tuple, Any 


class LemmaClassifierWithTransformer(nn.Module):

    def __init__(self, output_dim: int, model_type: str):
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

        # Get the embedding through transformer 

        # Choose transformer
        if model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.transformer = BertModel.from_pretrained('bert-base-uncased')
        elif model_type == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.transformer = RobertaModel.from_pretrained('roberta-base')
        config = self.transformer.config    

        embedding_size = config.hidden_size

        # define an MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, text: List[str], pos_index: int):
        """

        Args:
            text (List[str]): A single sentence with each token as an entry in the list.
            pos_index (int): The index of the token to classify on.

        Returns the logits of the MLP
        """

        # Get the transformer embeddings 
        input_ids = self.tokenizer.convert_tokens_to_ids(text)

        # Convert tokens to IDs and put them into a tensor
        input_ids_tensor = torch.tensor([input_ids])

        # Forward pass through Transformer
        with torch.no_grad():
            outputs = self.transformer(input_ids_tensor)
        
        # Get embeddings for all tokens
        last_hidden_state = outputs.last_hidden_state
        token_embeddings = last_hidden_state[0]

        # Get target embedding
        target_pos_embedding = token_embeddings[pos_index]

        # pass to the MLP
        output = self.mlp(target_pos_embedding)
        return output
