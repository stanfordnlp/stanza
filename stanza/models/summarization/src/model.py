import torch
import torch.nn as nn
import os
import logging
import math 
import torch.nn.functional as F


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
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text):
        outputs, (hidden, cell) = self.lstm(text)
        
        # Concatenate forward and backward reps
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)

        hidden_linearized = self.hidden_out(hidden)   # apply linear layer to reduce dimensionality -> initial decoder state
        cell_linearized = self.hidden_out(cell)

        return outputs, hidden, cell, hidden_linearized, cell_linearized


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W_h = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.W_s = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        self.b_attn = nn.Parameter(torch.rand(decoder_hidden_dim))

    
    def forward(self, encoder_outputs, decoder_hidden):

        seq_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # Repeat decoder hidden state seq_len times
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Compute energy scores with bias term
        energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden) + self.b_attn)
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class BaselineDecoder(nn.Module):

    def __init__(self, output_dim, encoder_hidden_dim, decoder_hidden_dim, num_layers=1, use_pgen=False):
        super(BaselineDecoder, self).__init__()

        self.output_dim = output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim

        self.embedding = nn.Embedding(output_dim, decoder_hidden_dim)  # this should be taken from the main model, also the args are wrong 


        self.lstm = nn.LSTM(decoder_hidden_dim + encoder_hidden_dim * 2, decoder_hidden_dim, num_layers=num_layers)   #  TODO : The input size is just the embedding size
        self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim)
        
        self.b, self.b_prime = nn.Parameter(torch.rand(decoder_hidden_dim)), nn.Parameter(torch.rand(output_dim))   # bias terms
        
        self.pgen = use_pgen

        if self.pgen:
            self.p_gen_linear = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, 1)     # TODO also add emb dim to this layer
         
        # Two linear layers as per equation (4) in the paper
        self.V = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim)
        self.V_prime = nn.Linear(decoder_hidden_dim, output_dim)
        
        self.softmax = nn.Softmax(dim=1)  # Softmax layer for the final output

    def forward(self, input, hidden, cell, encoder_outputs, src=None):
        
        # input is the current input token, which may be from the target sequence (teacher forcing)
        # or the previously predicted token. Initially, it could be a start-of-sequence token.
        embedded = self.embedding(input).unsqueeze(0)

        # Attention is computed using the decoder's current hidden state 'hidden' and all the encoder outputs
        attention_weights = self.attention(encoder_outputs, hidden)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs.transpose(0, 1))
        context_vector = context_vector.transpose(0, 1)

        # Here 'hidden' is the current state of the decoder, also known as decoder state
        # 'lstm_output' is the output of the LSTM at the current step, which can sometimes be different from 'hidden'
        # especially when using LSTM cells, since 'lstm_output' may be the output from the top layer of a multi-layer LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        # 'hidden' is now the updated decoder state after processing the current input token
        # This 'hidden' will be used in the next time step's attention computation

        p_gen = None
        if self.pgen:
            p_gen_input = torch.cat((context_vector, hidden.squeeze(0), embedded.squeeze(0)), dim=1)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))


        # The paper states that the decoder state (hidden) and the context vector are concatenated
        # before being passed through linear layers to predict the next token.
        concatenated = torch.cat((hidden.squeeze(0), context_vector.squeeze(0)), dim=1)
        output = self.V(concatenated) + self.b
        output = self.V_prime(output) + self.b_prime

        p_vocab = self.softmax(output)

        return p_vocab, hidden.squeeze(0), cell.squeeze(0), attention_weights, p_gen


class BaselineSeq2Seq(nn.Module):

    def __init__(self, encoder, decoder, vocab_size, embedding_dim):
        super(BaselineSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.pgen = self.decoder.pgen

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)

    def forward(self, text, target, teacher_forcing_ratio=0.5):
        batch_size = text.shape[1]  # for later
        trg_len = target.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(trg_len, trg_vocab_size)

        # Get embeddings over the text
        embedded = self.embedding(text)

        # Embeddings fed into Encoder LSTM
        # Get the hidden states h_i from the encoder
        # Take the encoder's hidden state after being passed through the linear layer into the decoder for the initial state
        encoder_outputs, encoder_hidden, encoder_cell, decoder_init_state, decoder_init_cell = self.encoder(embedded)

        # For each decoder time step, the decoder receives the word embedding of the previous word 
        input = target[0] # First input to the decoder is the <sos> token   TODO make sure the data uses <sos> at the beginning of sentences

        hidden = decoder_init_state   # the initial decoder hidden state is the linearized hidden state of the encoder
        cell = decoder_init_cell   # similar for the cell ^

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and cell states, and all encoder outputs
            # receive output tensor (predictions) and new hidden and cell states
            p_vocab, hidden, cell, attn_weights, pgen = self.decoder(input, hidden, cell, encoder_hidden)

            # if no pgen, then our final dist is p_vocab. otherwise, calculate the final distribution

            if self.pgen:   # TODO: figure this section out
                # p_vocab is the vocab distribution which should be shape (vocab_size, )
                # attn_weights is the attention distribution over the src text  (src_size, )
                p_vocab_scaled = pgen * p_vocab
                attn_dist_scaled = (1 - pgen) * attn_weights

                # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
                extended_vocab_size = self.vocab_size #  TODO must add the number of unknown OOV words to this

                extended_vocab_dist = torch.zeros(extended_vocab_size)
                extended_vocab_dist[: self.vocab_size] = p_vocab_scaled  # add the p_vocab to the fixed vocab

                # Add the attention probabilities (the probability of copying words from the source)
                # `src_extended_indices` is the tensor with the mapped indices of source words in the extended vocabulary
                # extended_vocab_dist.scatter_add_(1, src_extended_indices, attn_dist_scaled)
                # TODO: This extended vocab dist should become the logits for the output


            # Place predictions in a tensor holding predictions for each token
            outputs[t] = p_vocab

            # Decide whether to use teacher forcing or not
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            
            # Get the highest predicted token from our predictions
            top1 = p_vocab.argmax(1)

            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = target[t] if teacher_force else top1

        return outputs
