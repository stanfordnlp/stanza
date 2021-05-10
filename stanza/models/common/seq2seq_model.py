"""
The full encoder-decoder model, built on top of the base seq2seq modules.
"""

import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import stanza.models.common.seq2seq_constant as constant
from stanza.models.common import utils
from stanza.models.common.seq2seq_modules import LSTMAttention
from stanza.models.common.beam import Beam

logger = logging.getLogger('stanza')

class Seq2SeqModel(nn.Module):
    """
    A complete encoder-decoder model, with optional attention.
    """
    def __init__(self, args, emb_matrix=None, use_cuda=False):
        super().__init__()
        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']
        self.hidden_dim = args['hidden_dim']
        self.nlayers = args['num_layers'] # encoder layers, decoder layers = 1
        self.emb_dropout = args.get('emb_dropout', 0.0)
        self.dropout = args['dropout']
        self.pad_token = constant.PAD_ID
        self.max_dec_len = args['max_dec_len']
        self.use_cuda = use_cuda
        self.top = args.get('top', 1e10)
        self.args = args
        self.emb_matrix = emb_matrix

        logger.debug("Building an attentional Seq2Seq model...")
        logger.debug("Using a Bi-LSTM encoder")
        self.num_directions = 2
        self.enc_hidden_dim = self.hidden_dim // 2
        self.dec_hidden_dim = self.hidden_dim

        self.use_pos = args.get('pos', False)
        self.pos_dim = args.get('pos_dim', 0)
        self.pos_vocab_size = args.get('pos_vocab_size', 0)
        self.pos_dropout = args.get('pos_dropout', 0)
        self.edit = args.get('edit', False)
        self.num_edit = args.get('num_edit', 0)
        self.copy = args.get('copy', False)

        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.drop = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_token)
        self.encoder = nn.LSTM(self.emb_dim, self.enc_hidden_dim, self.nlayers, \
                bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.decoder = LSTMAttention(self.emb_dim, self.dec_hidden_dim, \
                batch_first=True, attn_type=self.args['attn_type'])
        self.dec2vocab = nn.Linear(self.dec_hidden_dim, self.vocab_size)
        if self.use_pos and self.pos_dim > 0:
            logger.debug("Using POS in encoder")
            self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_dim, self.pad_token)
            self.pos_drop = nn.Dropout(self.pos_dropout)
        if self.edit:
            edit_hidden = self.hidden_dim//2
            self.edit_clf = nn.Sequential(
                    nn.Linear(self.hidden_dim, edit_hidden),
                    nn.ReLU(),
                    nn.Linear(edit_hidden, self.num_edit))

        if self.copy:
            self.copy_gate = nn.Linear(self.dec_hidden_dim, 1)

        self.SOS_tensor = torch.LongTensor([constant.SOS_ID])
        self.SOS_tensor = self.SOS_tensor.cuda() if self.use_cuda else self.SOS_tensor

        self.init_weights()

    def init_weights(self):
        # initialize embeddings
        init_range = constant.EMB_INIT_RANGE
        if self.emb_matrix is not None:
            if isinstance(self.emb_matrix, np.ndarray):
                self.emb_matrix = torch.from_numpy(self.emb_matrix)
            assert self.emb_matrix.size() == (self.vocab_size, self.emb_dim), \
                    "Input embedding matrix must match size: {} x {}".format(self.vocab_size, self.emb_dim)
            self.embedding.weight.data.copy_(self.emb_matrix)
        else:
            self.embedding.weight.data.uniform_(-init_range, init_range)
        # decide finetuning
        if self.top <= 0:
            logger.debug("Do not finetune embedding layer.")
            self.embedding.weight.requires_grad = False
        elif self.top < self.vocab_size:
            logger.debug("Finetune top {} embeddings.".format(self.top))
            self.embedding.weight.register_hook(lambda x: utils.keep_partial_grad(x, self.top))
        else:
            logger.debug("Finetune all embeddings.")
        # initialize pos embeddings
        if self.use_pos:
            self.pos_embedding.weight.data.uniform_(-init_range, init_range)

    def cuda(self):
        super().cuda()
        self.use_cuda = True

    def cpu(self):
        super().cpu()
        self.use_cuda = False

    def zero_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = torch.zeros(self.encoder.num_layers*2, batch_size, self.enc_hidden_dim, requires_grad=False)
        c0 = torch.zeros(self.encoder.num_layers*2, batch_size, self.enc_hidden_dim, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        return h0, c0

    def encode(self, enc_inputs, lens):
        """ Encode source sequence. """
        h0, c0 = self.zero_state(enc_inputs)

        packed_inputs = nn.utils.rnn.pack_padded_sequence(enc_inputs, lens, batch_first=True)
        packed_h_in, (hn, cn) = self.encoder(packed_inputs, (h0, c0))
        h_in, _ = nn.utils.rnn.pad_packed_sequence(packed_h_in, batch_first=True)
        hn = torch.cat((hn[-1], hn[-2]), 1)
        cn = torch.cat((cn[-1], cn[-2]), 1)
        return h_in, (hn, cn)

    def decode(self, dec_inputs, hn, cn, ctx, ctx_mask=None, src=None):
        """ Decode a step, based on context encoding and source context states."""
        dec_hidden = (hn, cn)
        decoder_output = self.decoder(dec_inputs, dec_hidden, ctx, ctx_mask, return_logattn=self.copy)
        if self.copy:
            h_out, dec_hidden, log_attn = decoder_output
        else:
            h_out, dec_hidden = decoder_output

        h_out_reshape = h_out.contiguous().view(h_out.size(0) * h_out.size(1), -1)
        decoder_logits = self.dec2vocab(h_out_reshape)
        decoder_logits = decoder_logits.view(h_out.size(0), h_out.size(1), -1)
        log_probs = self.get_log_prob(decoder_logits)

        if self.copy:
            copy_logit = self.copy_gate(h_out)
            if self.use_pos:
                # can't copy the UPOS
                log_attn = log_attn[:, :, 1:]

            # renormalize
            log_attn = torch.log_softmax(log_attn, -1)
            # calculate copy probability for each word in the vocab
            log_copy_prob = torch.nn.functional.logsigmoid(copy_logit) + log_attn
            # scatter logsumexp
            mx = log_copy_prob.max(-1, keepdim=True)[0]
            log_copy_prob = log_copy_prob - mx
            copy_prob = torch.exp(log_copy_prob)
            copied_vocab_prob = log_probs.new_zeros(log_probs.size()).scatter_add(-1,
                src.unsqueeze(1).expand(src.size(0), copy_prob.size(1), src.size(1)),
                copy_prob)
            zero_mask = (copied_vocab_prob == 0)
            log_copied_vocab_prob = torch.log(copied_vocab_prob.masked_fill(zero_mask, 1e-12)) + mx
            log_copied_vocab_prob = log_copied_vocab_prob.masked_fill(zero_mask, -1e12)

            # combine with normal vocab probability
            log_nocopy_prob = -torch.log(1 + torch.exp(copy_logit))
            log_probs = log_probs + log_nocopy_prob
            log_probs = torch.logsumexp(torch.stack([log_copied_vocab_prob, log_probs]), 0)

        return log_probs, dec_hidden

    def forward(self, src, src_mask, tgt_in, pos=None):
        # prepare for encoder/decoder
        batch_size = src.size(0)
        enc_inputs = self.emb_drop(self.embedding(src))
        dec_inputs = self.emb_drop(self.embedding(tgt_in))
        if self.use_pos:
            assert pos is not None, "Missing POS input for seq2seq lemmatizer."
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(0).long().sum(1))

        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)

        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None

        log_probs, _ = self.decode(dec_inputs, hn, cn, h_in, src_mask, src=src)
        return log_probs, edit_logits

    def get_log_prob(self, logits):
        logits_reshape = logits.view(-1, self.vocab_size)
        log_probs = F.log_softmax(logits_reshape, dim=1)
        if logits.dim() == 2:
            return log_probs
        return log_probs.view(logits.size(0), logits.size(1), logits.size(2))

    def predict_greedy(self, src, src_mask, pos=None):
        """ Predict with greedy decoding. """
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, "Missing POS input for seq2seq lemmatizer."
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(constant.PAD_ID).long().sum(1))

        # encode source
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)

        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None

        # greedy decode by step
        dec_inputs = self.embedding(self.SOS_tensor)
        dec_inputs = dec_inputs.expand(batch_size, dec_inputs.size(0), dec_inputs.size(1))

        done = [False for _ in range(batch_size)]
        total_done = 0
        max_len = 0
        output_seqs = [[] for _ in range(batch_size)]

        while total_done < batch_size and max_len < self.max_dec_len:
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask, src=src)
            assert log_probs.size(1) == 1, "Output must have 1-step of output."
            _, preds = log_probs.squeeze(1).max(1, keepdim=True)
            dec_inputs = self.embedding(preds) # update decoder inputs
            max_len += 1
            for i in range(batch_size):
                if not done[i]:
                    token = preds.data[i][0].item()
                    if token == constant.EOS_ID:
                        done[i] = True
                        total_done += 1
                    else:
                        output_seqs[i].append(token)
        return output_seqs, edit_logits

    def predict(self, src, src_mask, pos=None, beam_size=5):
        """ Predict with beam search. """
        if beam_size == 1:
            return self.predict_greedy(src, src_mask, pos=pos)

        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, "Missing POS input for seq2seq lemmatizer."
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(constant.PAD_ID).long().sum(1))

        # (1) encode source
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)

        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None

        # (2) set up beam
        with torch.no_grad():
            h_in = h_in.data.repeat(beam_size, 1, 1) # repeat data for beam search
            src_mask = src_mask.repeat(beam_size, 1)
            # repeat decoder hidden states
            hn = hn.data.repeat(beam_size, 1)
            cn = cn.data.repeat(beam_size, 1)
        beam = [Beam(beam_size, self.use_cuda) for _ in range(batch_size)]

        def update_state(states, idx, positions, beam_size):
            """ Select the states according to back pointers. """
            for e in states:
                br, d = e.size()
                s = e.contiguous().view(beam_size, br // beam_size, d)[:,idx]
                s.data.copy_(s.data.index_select(0, positions))

        # (3) main loop
        for i in range(self.max_dec_len):
            dec_inputs = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1, 1)
            dec_inputs = self.embedding(dec_inputs)
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask, src=src)
            log_probs = log_probs.view(beam_size, batch_size, -1).transpose(0,1)\
                    .contiguous() # [batch, beam, V]

            # advance each beam
            done = []
            for b in range(batch_size):
                is_done = beam[b].advance(log_probs.data[b])
                if is_done:
                    done += [b]
                # update beam state
                update_state((hn, cn), b, beam[b].get_current_origin(), beam_size)

            if len(done) == batch_size:
                break

        # back trace and find hypothesis
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[0]]
            k = ks[0]
            hyp = beam[b].get_hyp(k)
            hyp = utils.prune_hyp(hyp)
            hyp = [i.item() for i in hyp]
            all_hyp += [hyp]

        return all_hyp, edit_logits

