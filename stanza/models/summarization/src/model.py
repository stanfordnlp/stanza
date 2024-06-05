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
from copy import deepcopy

logger = logging.getLogger('stanza.lemmaclassifier')

"""
Overall structure

Embedding layer
Bi-LSTM Encoder
Uni-LSTM Decoder

"""


class BaselineEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1, device=None, dropout_p: float = 0.5):
        super(BaselineEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, device=device)
        self.dropout = nn.Dropout(dropout_p).to(device)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim, device=device)
        self.device = device

    def forward(self, text):
        outputs, (hidden, cell) = self.lstm(text)
        unpacked_lstm_outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        unpacked_lstm_outputs = self.dropout(unpacked_lstm_outputs)
        
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).to(self.device) 
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).to(self.device)


        hidden_linearized = self.hidden_out(hidden)   # apply linear layer to reduce dimensionality -> initial decoder state
        cell_linearized = self.hidden_out(cell)
        return unpacked_lstm_outputs, hidden, cell, hidden_linearized, cell_linearized


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, coverage=False, device=None):
        super(BahdanauAttention, self).__init__()
        self.coverage = coverage  # use coverage or not
        self.device = device 

        self.W_h = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim, device=device)
        self.W_s = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, device=device)
        self.v = nn.Linear(decoder_hidden_dim, 1, device=device)
        self.b_attn = nn.Parameter(torch.rand(decoder_hidden_dim)).to(device)

        self.W_c = None
        if self.coverage:
            # self.W_c = nn.Linear(17, decoder_hidden_dim)  # replace 17 with seqlen, or maybe this should be max dec steps
            # self.W_c = nn.Conv1d(in_channels=1, out_channels=decoder_hidden_dim, kernel_size=1, bias=False)
            self.W_c = nn.Linear(1, decoder_hidden_dim, device=device)

    
    def forward(self, encoder_outputs, decoder_hidden, coverage_vec=None):

        """
        encoder_outputs: the hidden states from the encoder representation (batch size, seq length, 2 * enc hidden dim)
        decoder_hidden: the decoder hidden state to use for computing this attention. (batch size, decoder hidden dim)
        coverage_vec (optional): if coverage is enabled, this is the coverage vector for the decoder timestep. Should be
        shape (batch size, seq length).
        """

        seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]

        # Repeat decoder hidden state seq_len times to match enc states length
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1).to(self.device)

        # Compute energy scores with bias term

        """
        For a single batch example:

        the energy vector at timestep t, e^t, has shape (sequence length).
        the i-th element of the energy vector, e_i^t, is a scalar computed with c_i^t, also a scalar.
        this implies that the coverage vector at timestep t, c^t, has shape (sequence length)

        Therefore, we want to design the w_c term to have size (hidden dim), such that multiplying
        the two gives us shape (sequence length, hidden dim) which might require unsqueezing the w_c
        vector along the sequence length direction so that at the i-th hidden state, we have the same w_c.
        So what we really want is a tensor of shape (hidden dim, sequence length) for the W_c

        We want out (batch size, sequence length, hidden dim). So our input should be shape
        (batch size, sequence length, sequence length). We need to copy the dim of the coverage vec. 

        
        """
        energy = None
        if self.coverage:
            # unsqueeze the coverage vec to align it for the transformation to shape (batch size, seq len, decoder hidden dim) with W_c
            coverage_vec_unsqueezed = coverage_vec.unsqueeze(-1)
            energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden) + self.W_c(coverage_vec_unsqueezed) + self.b_attn).to(self.device)
        else:
            # no coverage, energy alignment scores become e_i^t = v^T tanh(W_h h_i + W_s s_t + b_attn)
            energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden) + self.b_attn).to(self.device)

        attention_raw = self.v(energy)
        attention = attention_raw.squeeze(2)

        # Generate mask: valid tokens have a nonzero hidden state
        mask = (encoder_outputs.abs().sum(dim=2) > 0).float().to(self.device)

        # Apply the mask by setting invalid positions to -inf
        attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(attention, dim=1)

        if self.coverage and coverage_vec is not None:
            new_coverage_vec = coverage_vec + attention   # continuously sum attention dist. for coverage
        else:
            new_coverage_vec = None
        return attention, new_coverage_vec


class BaselineDecoder(nn.Module):

    def __init__(self, output_dim, encoder_hidden_dim, decoder_hidden_dim, emb_dim, num_layers=1, use_pgen=False, use_coverage=False, device=None, dropout_p: float = 0.5):
        super(BaselineDecoder, self).__init__()

        self.output_dim = output_dim
        self.device = device
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim

        self.lstm = nn.LSTM(emb_dim, decoder_hidden_dim, num_layers=num_layers, batch_first=True, device=device)   
        self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim, coverage=use_coverage, device=device)
                
        self.pgen = use_pgen
        self.coverage = use_coverage

        if self.pgen:
            self.p_gen_linear = nn.Linear(encoder_hidden_dim * 2 + emb_dim + decoder_hidden_dim, 1, device=device) 

        if self.coverage:
            self.coverage_vec = None
         
        # Two linear layers as per equation (4) in the paper
        self.V_1 = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim, device=device)
        self.V_1_prime = nn.Linear(decoder_hidden_dim, output_dim, device=device)

        self.dropout = nn.Dropout(dropout_p).to(device)
        
        self.softmax = nn.LogSoftmax(dim=1)  # Softmax layer for the final output

    def forward(self, input, hidden, cell, encoder_outputs, src=None):

        """
        input : (batch size, emb dim)
        hidden : (batch size, decoder hidden dim)
        cell : (batch size, decoder hidden dim)
        encoder_outputs (batch size, seq len, encoder hidden dim)
        """
        
        batch_size = input.shape[0]
        sequence_length = encoder_outputs.shape[1]  
        if self.coverage and self.coverage_vec is None:
            # The first coverage vector is set to all zeros
            self.coverage_vec = torch.zeros(batch_size, sequence_length, device=self.device)
        
        # Attention is computed using the decoder's current hidden state 'hidden' and all the encoder outputs
        attention_weights, coverage_vec = self.attention(encoder_outputs, hidden, coverage_vec=self.coverage_vec if self.coverage else None)
        if self.coverage:
            self.coverage_vec = coverage_vec.detach()

        # Context vector = sum_i {a_i^t * h_i} 
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)  # (batch size, 2 * encoder hidden dim)

        context_vector = self.dropout(context_vector)

        # Here 'hidden' is the current state of the decoder, also known as decoder state
        # 'lstm_output' is the output of the LSTM at the current step, which can sometimes be different from 'hidden'
        # especially when using LSTM cells, since 'lstm_output' may be the output from the top layer of a multi-layer LSTM
        input = input.unsqueeze(1)  # (batch size, 1, embedding dim)
        hidden = hidden.unsqueeze(1).transpose(0, 1)
        cell = cell.unsqueeze(1).transpose(0, 1)

        lstm_output, (hidden, cell) = self.lstm(input, (hidden, cell))

        hidden, cell = hidden.transpose(0, 1), cell.transpose(0, 1)
        hidden, cell = hidden.squeeze(1), cell.squeeze(1)  # 'hidden' is now the updated decoder state after processing the current input token
        # This 'hidden' will be used in the next time step's attention computation
        # hidden & cell shape (batch size, decoder hidden dim)

        p_gen = None
        if self.pgen:
            p_gen_input = torch.cat((context_vector, hidden, input.squeeze(1)), dim=1)  # (batch size, 2 * encoder hidden dim + decoder hidden dim + emb dim)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))  # (batch size, 1)

        # Decoder state vector (hidden) and the context vector are concatenated
        # before being passed through linear layers to predict the next token.
        concatenated = torch.cat((hidden, context_vector), dim=1)

        output = self.V_1(concatenated) 
        output = self.V_1_prime(output) 

        output = self.dropout(output)

        p_vocab = self.softmax(output)

        coverage_vector = None
        if self.coverage:
            coverage_vector = self.coverage_vec

        return p_vocab, hidden, cell, attention_weights, p_gen, coverage_vector


class BaselineSeq2Seq(nn.Module):
    def __init__(self, model_args, pt_embedding, device, 
                 use_charlm: bool = False, charlm_forward_file: str = None, charlm_backward_file: str = None):
        
        super(BaselineSeq2Seq, self).__init__()
        self.model_args = model_args
        self.device = device
        self.batch_size = self.model_args.get("batch_size", DEFAULT_BATCH_SIZE)
        self.unsaved_modules = []
        self.input_size = 0
        self.max_enc_steps = self.model_args.get("max_enc_steps", None)   # truncate articles to max_enc_steps tokens 
        self.max_dec_steps = self.model_args.get("max_dec_steps", None)   # truncate summaries to max_dec_steps tokens
        self.use_charlm = use_charlm
        
        # word embeddings
        emb_matrix = pt_embedding.emb   # have to load this in through file by using 'load_pretrain' helper
        self.vocab_size = emb_matrix.shape[0]
        self.word_embedding_dim = emb_matrix.shape[1]
        self.vocab = pt_embedding.vocab  # to get word from index, used in characterlm
        START_TOKEN, STOP_TOKEN = "<s>", "</s>"
        start_vector = torch.randn((1, self.word_embedding_dim))
        stop_vector = torch.randn((1, self.word_embedding_dim))
        extended_embeddings = torch.cat((torch.from_numpy(emb_matrix), start_vector, stop_vector), dim=0)

        # self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=False)   # freeze False because 'See et. al.' updates embeddings

        self.embedding = nn.Embedding(len(self.vocab) + 2, self.word_embedding_dim, _weight=extended_embeddings)
        self.vocab_map = {word.replace('\xa0', ' '): i for i, word in enumerate(pt_embedding.vocab)}
        self.vocab_map[START_TOKEN] = len(self.vocab_map)
        self.vocab_map[STOP_TOKEN] = len(self.vocab_map)
        self.vocab_size += 2 

        # charlm embeddings
        if self.use_charlm:
            if charlm_forward_file is None or not os.path.exists(charlm_forward_file):
                raise FileNotFoundError(f"Could not find forward character model: {charlm_forward_file}")
            if charlm_backward_file is None or not os.path.exists(charlm_backward_file):
                raise FileNotFoundError(f"Could not find backward character model: {charlm_backward_file}")
            self.add_unsaved_module("charmodel_forward", CharacterLanguageModel.load(charlm_forward_file))
            self.add_unsaved_module("charmodel_backward", CharacterLanguageModel.load(charlm_backward_file))

            self.input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()

        # extended vocab 
        self.ext_vocab_map = deepcopy(self.vocab_map)
        self.max_oov_words = 0

        self.input_size += self.word_embedding_dim

        encoder_hidden_dim = self.model_args.get("encoder_hidden_dim", DEFAULT_ENCODER_HIDDEN_DIM)
        encoder_num_layers = self.model_args.get("encoder_num_layers", DEFAULT_ENCODER_NUM_LAYERS)
        self.encoder = BaselineEncoder(self.input_size, encoder_hidden_dim, num_layers=encoder_num_layers, device=device)

        decoder_hidden_dim = self.model_args.get("decoder_hidden_dim", encoder_hidden_dim)   # default value should be same hidden dim as encoder
        decoder_num_layers = self.model_args.get("decoder_num_layers", encoder_num_layers)
        self.pgen = self.model_args.get("pgen", False)
        self.coverage = self.model_args.get("coverage", False)
        self.decoder = BaselineDecoder(self.vocab_size, encoder_hidden_dim, decoder_hidden_dim, self.input_size, decoder_num_layers, self.pgen, self.coverage,
                                       device=device)
    
    def get_text_embeddings(self, text: List[List[str]], max_steps: int = None):
        """
        Extracts the word embeddings over the input articles in 'text'.
        
        Args:
            text (List[List[str]]): Tokenized batch of articles of text. Each inner List[str] is a single article.
            max_steps (int, optional): A limit on the maximum number of tokens in the texts. Defaults to no limit.

        Returns a tensor of the padded embeddings over the inputs. Also returns the input lengths.
        """
        device = next(self.parameters()).device

        if max_steps is not None:  # truncate text
            text = [article[: max_steps] for article in text]

        # Get word embeddings
        token_ids, input_lengths = [], []
        for article in text:
            article_token_ids = torch.tensor([self.vocab_map.get(word.lower(), UNK_ID) for word in article], device=device)
            token_ids.append(article_token_ids)
            input_lengths.append(len(article_token_ids))
        padded_inputs = pad_sequence(token_ids, batch_first=True).to(device)
        embedded = self.embedding(padded_inputs).to(device)  # (batch size, seq len, word emb dim)

        # Optionally, build Char embedding reps too
        if self.use_charlm:
            char_reps_forward = self.charmodel_forward.build_char_representation(text)
            char_reps_backward = self.charmodel_backward.build_char_representation(text)

            char_reps_forward = pad_sequence(char_reps_forward, batch_first=True)
            char_reps_backward = pad_sequence(char_reps_backward, batch_first=True)

            embedded = torch.cat((embedded, char_reps_forward, char_reps_backward), 2)  # (batch size, seq len, word emb dim + char emb dim)
        return embedded, input_lengths
    
    def build_extended_vocab_map(self, src: List[List[str]]):
        """
        constructs the extended vocabulary map between source document words and their indices
        
        for each document, the extended vocabulary is the union of the src document words and the existing vocab

        returns size (batch size, vocab size + num oov words)


        Instead of the index tensor being all zeros, which is an issue because for index tensors that don't get
        filled all the way, you get zero, which can be interpreted as zero index later. 

        As long as the attn mask is computed correctly, it should be okay for the index tensor to be zeroes.
        If the attn is computed, then out of sequence words get 0 attention.
        """
        max_oov_words = 0

        device = next(self.parameters()).device

        index_tensor = []
        for batch_idx, document in enumerate(src):
            num_oov_words = 0
            doc_indexes = []
            for i, word in enumerate(document):
                vocab_idx = self.ext_vocab_map.get(word.lower()) 

                if vocab_idx is None:
                    # If we cannot find the current word, we add it to the extended vocab
                    self.ext_vocab_map[word.lower()] = len(self.ext_vocab_map)  # new slot
                    num_oov_words += 1

                revised_idx = self.ext_vocab_map.get(word.lower())
                if revised_idx is None:
                    raise ValueError(f"Error building extended vocab map, word: {word}")
                doc_indexes.append(revised_idx)

            index_tensor.append(torch.tensor(doc_indexes, device=device))
            max_oov_words = max(max_oov_words, num_oov_words)
        
        revised_index_tensor = pad_sequence(index_tensor, batch_first=True)
        return revised_index_tensor, max_oov_words

    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def forward(self, text, target, teacher_forcing_ratio=1.0):
        """
        text (List[List[str]]): The outer list is a collection of the examples used in training. The inner list is composed of the words within each example.
        e.g. [["The", "quick", "brown", "fox"], ["Humpty", "dumpty", "fell", "off", "the", "wall", "."]]
        This can be thought of to have shape (batch size, seq len) even though the inputs at this point aren't padded

        target (List[List[str]]): A list of reference summaries for the input texts. The outer list is a collection of the reference summaries.
        The inner list is composed of words within each reference summary.

        teacher_forcing_ratio (float, optional): Probability to use for teacher forcing method. Should be set to 0 during test time.

        """
        device = next(self.parameters()).device
        batch_size = min(len(text), self.batch_size) 

        if self.max_enc_steps is not None:  # truncate text
            text = [article[: self.max_enc_steps] for article in text]
        if self.max_dec_steps is not None:  # truncate target
            target = [summary[: self.max_dec_steps] for summary in target]

        index_tensor, max_oov_words = self.build_extended_vocab_map(text)  # (batch size, seq len)
        self.max_oov_words = max_oov_words   # the count of OOV words in the input texts 

        # Get embeddings over the input text
        embedded, input_lengths = self.get_text_embeddings(text, self.max_enc_steps)
        input_len = embedded.shape[1]  # the max seq len out of all inputs

        # Get embeddings over the target text
        target_embeddings, target_lengths = self.get_text_embeddings(target, self.max_dec_steps)  # (batch size, seq len, input size)
        
        target_len = target_embeddings.shape[1]   # TODO : Ask John how batch processing works with this. should this actually just be a uniform hyperparam like max_dec_steps? If so, how to do padding?
        # target_len currently represents the maximum seq length out of all sequences in the reference summaries.

        # Tensor to store decoder outputs
        effective_vocab_size = len(self.ext_vocab_map) if self.pgen else self.vocab_size

        packed_input_seqs = pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)

        # Embeddings fed into Encoder LSTM
        # Get the hidden states h_i from the encoder
        # Take the encoder's hidden state after being passed through the linear layer into the decoder for the initial state
        encoder_outputs, encoder_hidden, encoder_cell, decoder_init_state, decoder_init_cell = self.encoder(packed_input_seqs)

        # For each decoder time step, the decoder receives the word embedding of the previous word 
        input = target_embeddings[:, 0, :]

        hidden = decoder_init_state   # the initial decoder hidden state is the linearized hidden state of the encoder
        cell = decoder_init_cell   # similar for the cell ^

        # we want the attention weights return object to be shape (batch size, decoder timestep, input sequence length)
        # we want the coverage vector return object to be shape (batch size, decoder timestep, input sequence length)
        outputs = torch.zeros(batch_size, target_len, effective_vocab_size, device=device)
        final_attn_weights = torch.zeros(batch_size, target_len, input_len, device=device)
        final_coverage_vecs = torch.zeros(batch_size, target_len, input_len, device=device)

        for t in range(target_len):  # decoder timesteps
            p_vocab, hidden, cell, attn_weights, pgen, coverage_vector = self.decoder(input, hidden, cell, encoder_outputs)

            # attn weights is (batch size, input seq len)
            # coverage vector is (batch size, input seq len)

            # if no pgen, then our final dist is p_vocab. otherwise, calculate the final distribution
            if self.pgen:   
                p_vocab_scaled = pgen * p_vocab   # (batch size, vocab size)
                attn_dist_scaled = (1 - pgen) * attn_weights   # (batch size, seq len)

                """
                For each word in the sequence, we need to know if it is in the vocabulary or not. 
                If the word is is in the vocab, then its value in the extended vocabulary distribution 
                is p_gen * P_vocab(w) + (1 - p_gen) * sum_i a_i^t

                If the word is not in the vocab, then it receives a new index in the extended vocab and 
                its value is (1 - p_gen) * sum_i a_i^t

                The extended vocab will be shape (batch size, extended vocab size)
                So for the batches, it will be (batch size, max extended vocab size)
                We start with zeroes, copy over the scaled P_vocab distribution, and then add where appropriate?

                So, we need a tensor of shape (batch size, seq len) that gives each word's index in the vocab.
                Once we have the word's index in the vocab, we know which index to add the summation term to.
                For new words, we create a new index. We also need an extended vocab map.
                """

                # at this point, assume that we have a tensor where for each batch, the i-th index of the tensor
                # contains the index of the sequence token in the extended vocab

                # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
                extended_vocab_size = len(self.ext_vocab_map)  # vocab size + number of unique OOV words across all batch examples
                extended_vocab_dist = torch.zeros(batch_size, extended_vocab_size, device=device)   # one vocab dist per text
                extended_vocab_dist[:, :self.vocab_size] = p_vocab_scaled  # fill the extended vocab with the existing distribution we have

                # add the attention distribution to the extended vocab distribution to include input text words.
                final_vocab_dist = extended_vocab_dist.scatter_add(dim=1, index=index_tensor, src=attn_dist_scaled)
                p_vocab = final_vocab_dist

            # Place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = p_vocab
            final_attn_weights[:, t, :] = attn_weights
            if self.coverage:
                final_coverage_vecs[:, t, :] = coverage_vector

            # Decide whether to use teacher forcing or not
            teacher_force = torch.rand(1) < teacher_forcing_ratio

            # Get the highest predicted token from our predictions
            top1 = torch.argmax(p_vocab, dim=1)  # (batch size, 1)

            # If teacher forcing, use actual next token as next input. If not, use the predicted token.
            if teacher_force:
                input = target_embeddings[:, t, :].to(device)   # token for timestep t in reference summary
            else:
                # When an OOV word is chosen as the next token, we give the UNK embedding as the input embedding 
                # as the next word embedding because we do not have an embedding for this word
                oov_words_mask = top1 >= self.vocab_size  # masking which chosen words are out of vocabulary
                top1[oov_words_mask] = UNK_ID
                # generate embedding for next word
                input = self.embedding(top1) 
                if self.use_charlm:
                    # build_char_representation take [[str]], so we need to convert the tensor of IDs to list of strings
                    id2unit = {idx: word for word, idx in self.ext_vocab_map.items()}
                    top1_words = [id2unit.get(word.item()) for word in top1]
                    chosen_words = [[word] for word in top1_words]

                    char_reps_forward = self.charmodel_forward.build_char_representation(chosen_words)
                    char_reps_backward = self.charmodel_backward.build_char_representation(chosen_words)
                    
                    char_reps_forward = pad_sequence(char_reps_forward, batch_first=True)
                    char_reps_backward = pad_sequence(char_reps_backward, batch_first=True)

                    # the char reps are shape (batch size, seq len, char_dim) but the seq len is always 1 because we only choose 
                    # 1 token, so we simply squeeze the seqlen dim to get (batch size, char_dim) to concat with the word emb
                    input = torch.cat((input, char_reps_forward.squeeze(1), char_reps_backward.squeeze(1)), 1)  # (batch size, 1, word emb dim + char emb dim)
                
        self.ext_vocab_map = deepcopy(self.vocab_map)  # reset OOV words for the next batch of text
        self.max_oov_words = 0

        return outputs, final_attn_weights, final_coverage_vecs

    def run_encoder(self, examples, max_enc_steps: int = None):
        """
        For beam search decoding: run the encoder and return the encoder outputs, and the decoder init states
        
        examples: [[str]], the tokenized text of batches of article examples
        """
        embedded, input_lens = self.get_text_embeddings(examples, max_steps=max_enc_steps)
        packed_input_seqs = pack_padded_sequence(embedded, input_lens, batch_first=True, enforce_sorted=False)
        unpacked_lstm_outputs, hidden, cell, hidden_linearized, cell_linearized = self.encoder(packed_input_seqs)
        return unpacked_lstm_outputs, hidden_linearized, cell_linearized

    def decode_onestep(self, examples: List[List[str]], latest_tokens: List[List[str]], enc_states: torch.Tensor, 
                       dec_hidden: torch.Tensor, dec_cell: torch.Tensor, prev_coverage: torch.Tensor):
        """
        For beam search decoding: Run the decoder for a single step.

        Args:
            examples (List[List[str]]): the raw tokenized text of the batch examples
            latest_tokens (List[List[str]]): the tokens used as input to the decoder for this timestep (i.e. last token to be decoded)
            enc_states: The encoder states for the text (beam size, seq len, enc hidden dim)
            dec_hidden: A tensor of the hidden state of the decoder at the current timestep for each batch (beam size, dec hidden dim)
            dec_cell: A tensor of the cell state of the decoder at the current timestep for each batch (beam size, dec hidden dim)
            prev_coverage: Coverage vectors from the last timestep. None if not using coverage.

        Returns:
            ids: top 2k token ids from prediction (beam size, 2 * beam size)
            probs: top 2k log probabilities. shape (beam size, 2 * beam size)
            new_hidden: new hidden states for the decoder. shape (beam size, dec hidden dim)
            attn_dists: attention distributions. shape (beam size, seq len)
            p_gens: generation probabilities fro this step. shape (beam size)
            new_coverage: Coverage vectors for this step. Shape (beam size, seq len)

        TODO be careful that this function might set new state variables for the model components, so 
        be mindful of this and just load a model checkpoint when using this. Do not run on an in-progress
        model being trained

        """

        beam_size = dec_hidden.shape[0]
        device = next(self.parameters()).device
        index_tensor, max_oov_words = self.build_extended_vocab_map(examples)

        latest_token_emb, _ = self.get_text_embeddings(latest_tokens)  # batch size, seq len (1), emb dim
        latest_token_emb = latest_token_emb.squeeze(1)  # squeeze the seqlen dim

        attention_weights, coverage_vec = self.decoder.attention(enc_states, dec_hidden, prev_coverage)
        if prev_coverage is not None and self.coverage:  # remove from device because this updates
            prev_coverage = coverage_vec.detach()

        context_vector = torch.bmm(attention_weights.unsqueeze(1), enc_states)
        context_vector = context_vector.squeeze(1)  # (beam size, 2 * enc hidden dim)

        input = latest_token_emb.unsqueeze(1)  # (batch size, 1, embedding dim)
        dec_hidden = dec_hidden.unsqueeze(1).transpose(0, 1)
        dec_cell = dec_cell.unsqueeze(1).transpose(0, 1)

        lstm_output, (hidden, cell) = self.decoder.lstm(input, (dec_hidden, dec_cell))

        hidden, cell = hidden.transpose(0, 1), cell.transpose(0, 1)
        dec_hidden, dec_cell = hidden.squeeze(1), cell.squeeze(1)  # Updated decoder state (beam size, decoder hidden dim)

        p_gen = None
        if self.pgen:
            p_gen_input = torch.cat((context_vector, dec_hidden, input.squeeze(1)), dim=1)  # (beam size, 2 * enc_hidden_dim + dec hidden dim + emb dim)
            p_gen = torch.sigmoid(self.decoder.p_gen_linear(p_gen_input))  # (beam size, 1)
        
        # Decoder state and context vector and concatenated and passed through linear layers to produce the vocab dist.
        concatenated = torch.cat((dec_hidden, context_vector), dim=1)
        output = self.decoder.V_1(concatenated)
        output = self.decoder.V_1_prime(output)
        p_vocab = self.decoder.softmax(output)  # (beam size, vocab size)

        # Make final distribution if needed
        if self.pgen:
            scaled_p_vocab = p_gen * p_vocab  # (beam size, vocab size)
            attention_weights = (1 - p_gen) * attention_weights  # (beam size, seq len)

            extended_vocab_size = len(self.ext_vocab_map)
            extended_vocab_dist = torch.zeros(beam_size, extended_vocab_size, device=device) 
            extended_vocab_dist[:, :self.vocab_size] = scaled_p_vocab

            final_vocab_dist = extended_vocab_dist.scatter_add(dim=1, index=index_tensor, src=attention_weights)
            p_vocab = final_vocab_dist
        
        # Produce top 2k IDs and top 2k log probabilities
        top_k_probs, top_k_ids = torch.topk(p_vocab, 2*beam_size, dim=1)
        top_k_probs = torch.log(top_k_probs)
        return top_k_ids, top_k_probs, dec_hidden, dec_cell, attention_weights, p_gen, prev_coverage, self.ext_vocab_map
