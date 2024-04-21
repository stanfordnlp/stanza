import torch
import torch.nn as nn
import os
import logging
import math 


from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from typing import List, Tuple

logger = logging.getLogger('stanza.lemmaclassifier')


"""
Overall structure

Embedding layer
Bi-LSTM Encoder
Uni-LSTM Decoder

"""


class BaselineEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BaselineEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True)

    def forward(self, text):
        outputs, (hidden, cell) = self.lstm(text)
        
        # Concatenate forward and backward reps
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)

        return hidden, cell


class BaselineDecoder(nn.Module):

    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(BaselineDecoder, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(output_dim, hidden_dim * 2, num_layers=num_layers)   # hidden_dim * 2 because it takes the bi-LSTM encodings
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input.unsqueeze(0), (hidden.unsqueeze(0), cell.unsqueeze(0)))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden.squeeze(0), cell.squeeze(0)


class BaselineSeq2Seq(nn.Module):

    def __init__(self, encoder, decoder, vocab_size, embedding_dim):
        super(BaselineSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)

    def forward(self, text):

        pass 
