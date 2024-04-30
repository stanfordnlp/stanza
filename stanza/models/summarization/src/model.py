import torch
import torch.nn as nn
import os
import logging
import math 
import torch.nn.functional as F


from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from stanza.models.common.foundation_cache import load_pretrain

from stanza.models.summarization.constants import * 
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text):
        outputs, (hidden, cell) = self.lstm(text)
        unpacked_lstm_outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)


        hidden_linearized = self.hidden_out(hidden)   # apply linear layer to reduce dimensionality -> initial decoder state
        cell_linearized = self.hidden_out(cell)
        return unpacked_lstm_outputs, hidden, cell, hidden_linearized, cell_linearized


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W_h = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.W_s = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        self.b_attn = nn.Parameter(torch.rand(decoder_hidden_dim))

    
    def forward(self, encoder_outputs, decoder_hidden):
        seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]

        # Repeat decoder hidden state seq_len times
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Compute energy scores with bias term
        energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden) + self.b_attn)
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class BaselineDecoder(nn.Module):

    def __init__(self, output_dim, encoder_hidden_dim, decoder_hidden_dim, emb_dim, num_layers=1, use_pgen=False):
        super(BaselineDecoder, self).__init__()

        self.output_dim = output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim

        self.lstm = nn.LSTM(emb_dim, decoder_hidden_dim, num_layers=num_layers, batch_first=True)   
        self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim)
                
        self.pgen = use_pgen

        if self.pgen:
            self.p_gen_linear = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, 1)     # TODO also add emb dim to this layer
         
        # Two linear layers as per equation (4) in the paper
        self.V = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim)
        self.V_prime = nn.Linear(decoder_hidden_dim, output_dim)
        
        self.softmax = nn.Softmax(dim=1)  # Softmax layer for the final output

    def forward(self, input, hidden, cell, encoder_outputs, src=None):
        
        # Attention is computed using the decoder's current hidden state 'hidden' and all the encoder outputs
        attention_weights = self.attention(encoder_outputs, hidden)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)

        # Here 'hidden' is the current state of the decoder, also known as decoder state
        # 'lstm_output' is the output of the LSTM at the current step, which can sometimes be different from 'hidden'
        # especially when using LSTM cells, since 'lstm_output' may be the output from the top layer of a multi-layer LSTM
        input = input.unsqueeze(1)
        hidden = hidden.unsqueeze(1).transpose(0, 1)
        cell = cell.unsqueeze(1).transpose(0, 1)

        lstm_output, (hidden, cell) = self.lstm(input, (hidden, cell))

        hidden, cell = hidden.transpose(0, 1), cell.transpose(0, 1)
        hidden, cell = hidden.squeeze(1), cell.squeeze(1)

        # 'hidden' is now the updated decoder state after processing the current input token
        # This 'hidden' will be used in the next time step's attention computation

        p_gen = None
        if self.pgen:
            p_gen_input = torch.cat((context_vector, hidden.squeeze(0), input.squeeze(0)), dim=1)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))


        # The paper states that the decoder state (hidden) and the context vector are concatenated
        # before being passed through linear layers to predict the next token.
        concatenated = torch.cat((hidden.squeeze(0), context_vector.squeeze(0)), dim=1)

        output = self.V(concatenated) 
        output = self.V_prime(output) 

        p_vocab = self.softmax(output)

        return p_vocab, hidden.squeeze(0), cell.squeeze(0), attention_weights, p_gen


class BaselineSeq2Seq(nn.Module):
    """
    
    """
    def __init__(self, model_args, pt_embedding):
        super(BaselineSeq2Seq, self).__init__()
        self.model_args = model_args
        self.input_size = 0
        self.batch_size = self.model_args.get("batch_size", DEFAULT_BATCH_SIZE)
        
        emb_matrix = pt_embedding.emb   # have to load this in through file by using 'load_pretrain' helper
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=False)   # freeze False because 'See et. al.' updates embeddings
        self.vocab_map = {word.replace('\xa0', ' '): i for i, word in enumerate(pt_embedding.vocab)}

        self.input_size += self.embedding_dim

        encoder_hidden_dim = self.model_args.get("encoder_hidden_dim", DEFAULT_ENCODER_HIDDEN_DIM)
        encoder_num_layers = self.model_args.get("encoder_num_layers", DEFAULT_ENCODER_NUM_LAYERS)
        self.encoder = BaselineEncoder(self.input_size, encoder_hidden_dim, num_layers=encoder_num_layers)

        decoder_hidden_dim = self.model_args.get("decoder_hidden_dim", encoder_hidden_dim)   # default value should be same hidden dim as encoder
        decoder_num_layers = self.model_args.get("decoder_num_layers", encoder_num_layers)
        self.pgen = self.model_args.get("pgen", False)
        self.decoder = BaselineDecoder(self.vocab_size, encoder_hidden_dim, decoder_hidden_dim, self.embedding_dim, decoder_num_layers, self.pgen)
    

    def extract_word_embeddings(self, text: List[List[str]]):
        """
        Extracts the word embeddings over the input articles in 'text'.

        text (List[List[str]]): Tokenized articles of text
        """
        token_ids, input_lengths = [], []
        for article in text:
            article_token_ids = torch.tensor([self.vocab_map.get(word.lower(), UNK_ID) for word in article])
            token_ids.append(article_token_ids)
            input_lengths.append(len(article_token_ids))
        padded_inputs = pad_sequence(token_ids, batch_first=True)
        embedded = self.embedding(padded_inputs)
        return embedded, input_lengths

    def forward(self, text, target, teacher_forcing_ratio=0.5):
        batch_size = 2 # TODO later fix 
        target_len = 11  # TODO later fix

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, self.vocab_size)

        # Get embeddings over the input text
        embedded, input_lengths = self.extract_word_embeddings(text)

        # Get embeddings over the target text
        target_embeddings, target_lengths = self.extract_word_embeddings(target)

        packed_input_seqs = pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        # packed_target_seqs = pack_padded_sequence(target_embeddings, target_lengths, batch_first=True, enforce_sorted=False)

        # Embeddings fed into Encoder LSTM
        # Get the hidden states h_i from the encoder
        # Take the encoder's hidden state after being passed through the linear layer into the decoder for the initial state
        encoder_outputs, encoder_hidden, encoder_cell, decoder_init_state, decoder_init_cell = self.encoder(packed_input_seqs)

        # For each decoder time step, the decoder receives the word embedding of the previous word 
        input = target_embeddings[:, 0, :] # TODO make sure the data uses <sos> at the beginning of sentences

        hidden = decoder_init_state   # the initial decoder hidden state is the linearized hidden state of the encoder
        cell = decoder_init_cell   # similar for the cell ^

        for t in range(target_len):
            p_vocab, hidden, cell, attn_weights, pgen = self.decoder(input, hidden, cell, encoder_outputs)

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
            outputs[:, t, :] = p_vocab

            # Decide whether to use teacher forcing or not
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            
            # Get the highest predicted token from our predictions
            top1 = torch.argmax(p_vocab, dim=1)

            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = target_embeddings[:, t, :] if teacher_force else self.embedding(top1)
        return outputs
