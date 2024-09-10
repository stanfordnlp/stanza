"""
Classify characters based on an LSTM with learned character representations
"""

import logging

import torch
from torch import nn

import stanza.models.common.seq2seq_constant as constant

logger = logging.getLogger('stanza')

class CharacterClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']
        self.hidden_dim = args['hidden_dim']
        self.nlayers = args['num_layers'] # lstm encoder layers
        self.pad_token = constant.PAD_ID
        self.enc_hidden_dim = self.hidden_dim // 2   # since it is bidirectional

        self.num_outputs = 2

        self.args = args

        self.emb_dropout = args.get('emb_dropout', 0.0)
        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.dropout = args['dropout']

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_token)
        self.input_dim = self.emb_dim
        self.encoder = nn.LSTM(self.input_dim, self.enc_hidden_dim, self.nlayers, \
                               bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)

        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_outputs))

    def encode(self, enc_inputs, lens):
        """ Encode source sequence. """
        packed_inputs = nn.utils.rnn.pack_padded_sequence(enc_inputs, lens, batch_first=True)
        packed_h_in, (hn, cn) = self.encoder(packed_inputs)
        return packed_h_in

    def embed(self, src, src_mask):
        # the input data could have characters outside the known range
        # of characters in cases where the vocabulary was temporarily
        # expanded (note that this model does nothing with those chars)
        embed_src = src.clone()
        embed_src[embed_src >= self.vocab_size] = constant.UNK_ID
        enc_inputs = self.emb_drop(self.embedding(embed_src))
        batch_size = enc_inputs.size(0)
        src_lens = list(src_mask.data.eq(self.pad_token).long().sum(1))
        return enc_inputs, batch_size, src_lens, src_mask

    def forward(self, src, src_mask):
        enc_inputs, batch_size, src_lens, src_mask = self.embed(src, src_mask)
        encoded = self.encode(enc_inputs, src_lens)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        logits = self.output_layer(encoded)
        return logits
