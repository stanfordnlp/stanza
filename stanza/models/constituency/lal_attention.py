import functools

import numpy as np
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import transformers

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch_t = torch.cuda
    def from_numpy(ndarray):
        if float(sys.version[:3]) <= 3.6:
            return eval('torch.from_numpy(ndarray).pin_memory().cuda(async=True)')
        else:
            return torch.from_numpy(ndarray).pin_memory().cuda(non_blocking=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
#import src_dep_const_test.chart_helper as chart_helper
import hpsg_decoder
import const_decoder
import makehp
import utils

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
ROOT = "<START>"
Sub_Head = "<H>"
No_Head = "<N>"

DTYPE = torch.uint8 if float(sys.version[:3]) < 3.7 else torch.bool

TAG_UNK = "UNK"

ROOT_TYPE = "<ROOT_TYPE>"

# Assumes that these control characters are not present in treebank text
CHAR_UNK = "\0"
CHAR_START_SENTENCE = "\1"
CHAR_START_WORD = "\2"
CHAR_STOP_WORD = "\3"
CHAR_STOP_SENTENCE = "\4"
CHAR_PAD = "\5"

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--", # en dash
    "\u2014": "--", # em dash
    }


class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """
    def __init__(self, batch_idxs_np):
        self.batch_idxs_np = batch_idxs_np
        self.batch_idxs_torch = from_numpy(batch_idxs_np)

        self.batch_size = int(1 + np.max(batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))

#
class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, batch_idxs, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = input.new().resize_(batch_idxs.batch_size, input.size(1))
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None, None
        else:
            return grad_output, None, None, None, None

#
class FeatureDropout(nn.Module):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input, batch_idxs):
        return FeatureDropoutFunction.apply(input, batch_idxs, self.p, self.training, self.inplace)

#
class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

#
class ScaledAttention(nn.Module):
    def __init__(self, hparams, attention_dropout=0.1):
        super(ScaledAttention, self).__init__()
        self.hparams = hparams
        self.temper = hparams.d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat]
        # k: [batch, slot, feat]
        # v: [batch, slot, feat]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper


        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn.transpose(1, 2)).transpose(1, 2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

# %%

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat] or (batch * d_l) x max_len x d_k
        # k: [batch, slot, feat] or (batch * d_l) x max_len x d_k
        # v: [batch, slot, feat] or (batch * d_l) x max_len x d_v
        # q in LAL is (batch * d_l) x 1 x d_k

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper # (batch * d_l) x max_len x max_len
        # in LAL, gives: (batch * d_l) x 1 x max_len
        # attention weights from each word to each word, for each label
        # in best model (repeated q): attention weights from label (as vector weights) to each word

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        # Note that this makes the distribution not sum to 1. At some point it
        # may be worth researching whether this is the right way to apply
        # dropout to the attention.
        # Note that the t2t code also applies dropout in this manner
        attn = self.dropout(attn)
        output = torch.bmm(attn, v) # (batch * d_l) x max_len x d_v
        # in LAL, gives: (batch * d_l) x 1 x d_v

        return output, attn

#
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(self, hparams, n_head, d_model, d_k, d_v, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.hparams = hparams

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            self.w_qs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_ks1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_vs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_v // 2))

            self.w_qs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_ks2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_vs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_v // 2))

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            self.w_qs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_v))

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            self.proj = nn.Linear(n_head*d_v, d_model, bias=False)
        else:
            self.proj1 = nn.Linear(n_head*(d_v//2), self.d_content, bias=False)
            self.proj2 = nn.Linear(n_head*(d_v//2), self.d_positional, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, qk_inp=None):
        v_inp_repeated = inp.repeat(self.n_head, 1).view(self.n_head, -1, inp.size(-1)) # n_head x len_inp x d_model
        if qk_inp is None:
            qk_inp_repeated = v_inp_repeated
        else:
            qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, -1, qk_inp.size(-1))

        if not self.partitioned:
            q_s = torch.bmm(qk_inp_repeated, self.w_qs) # n_head x len_inp x d_k
            k_s = torch.bmm(qk_inp_repeated, self.w_ks) # n_head x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs) # n_head x len_inp x d_v
        else:
            q_s = torch.cat([
                torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_qs1),
                torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_qs2),
                ], -1)
            k_s = torch.cat([
                torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_ks1),
                torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_ks2),
                ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:,:,self.d_content:], self.w_vs2),
                ], -1)
        return q_s, k_s, v_s

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * mb_size) x len_padded x d
        # (along with masks for the attention and output)
        n_head = self.n_head
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=DTYPE)

        for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
            k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
            v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
            invalid_mask[i, :end-start].fill_(False)

        return(
            q_padded.view(-1, len_padded, d_k),
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1),
            (~invalid_mask).repeat(n_head, 1),
            )

    def combine_v(self, outputs):
        # Combine attention information from the different heads
        n_head = self.n_head
        outputs = outputs.view(n_head, -1, self.d_v) # n_head x len_inp x d_kv

        if not self.partitioned:
            # Switch from n_head x len_inp x d_v to len_inp x (n_head * d_v)
            outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)

            # Project back to residual size
            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:,:,:d_v1]
            outputs2 = outputs[:,:,d_v1:]
            outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
                ], -1)

        return outputs

    def forward(self, inp, batch_idxs, qk_inp=None):
        residual = inp

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, qk_inp=qk_inp)
        # n_head x len_inp x d_kv

        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)
        # (n_head * batch) x len_padded x d_kv

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask,
            )
        outputs = outputs_padded[output_mask]
        # (n_head * len_inp) x d_kv
        outputs = self.combine_v(outputs)
        # len_inp x d_model

        outputs = self.residual_dropout(outputs, batch_idxs)

        return self.layer_norm(outputs + residual), attns_padded

#
class PositionwiseFeedForward(nn.Module):
    """
    A position-wise feed forward module.

    Projects to a higher-dimensional space before applying ReLU, then projects
    back.
    """

    def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_ff)
        self.w_2 = nn.Linear(d_ff, d_hid)

        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()


    def forward(self, x, batch_idxs):
        residual = x

        output = self.w_1(x)
        output = self.relu_dropout(self.relu(output), batch_idxs)
        output = self.w_2(output)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

#
class PartitionedPositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_ff, d_positional, relu_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff//2)
        self.w_1p = nn.Linear(d_positional, d_ff//2)
        self.w_2c = nn.Linear(d_ff//2, self.d_content)
        self.w_2p = nn.Linear(d_ff//2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x
        xc = x[:, :self.d_content]
        xp = x[:, self.d_content:]

        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)

        output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

#
class MultiLevelEmbedding(nn.Module):
    def __init__(self,
            num_embeddings_list,
            d_embedding,
            hparams,
            d_positional=None,
            max_len=300,
            normalize=True,
            dropout=0.1,
            timing_dropout=0.0,
            emb_dropouts_list=None,
            extra_content_dropout=None,
            word_table_np = None,
            **kwargs):
        super().__init__()

        self.d_embedding = d_embedding
        self.partitioned = d_positional is not None
        self.hparams = hparams

        if self.partitioned:
            self.d_positional = d_positional
            self.d_content = self.d_embedding - self.d_positional
        else:
            self.d_positional = self.d_embedding
            self.d_content = self.d_embedding

        if emb_dropouts_list is None:
            emb_dropouts_list = [0.0] * len(num_embeddings_list)
        assert len(emb_dropouts_list) == len(num_embeddings_list)

        if word_table_np is not None:
            self.pretrain_dim = word_table_np.shape[1]
        else:
            self.pretrain_dim = 0

        embs = []
        emb_dropouts = []
        cun = len(num_embeddings_list)*2
        for i, (num_embeddings, emb_dropout) in enumerate(zip(num_embeddings_list, emb_dropouts_list)):
            if hparams.use_cat:
                if i == len(num_embeddings_list) - 1:
                    #last is word
                    emb = nn.Embedding(num_embeddings, self.d_content//cun - self.pretrain_dim, **kwargs)
                else :
                    emb = nn.Embedding(num_embeddings, self.d_content//cun, **kwargs)
            else :
                emb = nn.Embedding(num_embeddings, self.d_content - self.pretrain_dim, **kwargs)
            embs.append(emb)
            emb_dropout = FeatureDropout(emb_dropout)
            emb_dropouts.append(emb_dropout)

        if word_table_np is not None:
            self.pretrain_emb = nn.Embedding(word_table_np.shape[0], self.pretrain_dim)
            self.pretrain_emb.weight.data.copy_(torch.from_numpy(word_table_np))
            self.pretrain_emb.weight.requires_grad_(False)
            self.pretrain_emb_dropout = FeatureDropout(0.33)

        self.embs = nn.ModuleList(embs)
        self.emb_dropouts = nn.ModuleList(emb_dropouts)

        if extra_content_dropout is not None:
            self.extra_content_dropout = FeatureDropout(extra_content_dropout)
        else:
            self.extra_content_dropout = None

        if normalize:
            self.layer_norm = LayerNormalization(d_embedding)
        else:
            self.layer_norm = lambda x: x

        self.dropout = FeatureDropout(dropout)
        self.timing_dropout = FeatureDropout(timing_dropout)

        # Learned embeddings
        self.max_len = max_len
        self.position_table = nn.Parameter(torch_t.FloatTensor(max_len, self.d_positional))
        init.normal_(self.position_table)

    def forward(self, xs, pre_words_idxs, batch_idxs, extra_content_annotations=None):
        content_annotations = [
            emb_dropout(emb(x), batch_idxs)
            for x, emb, emb_dropout in zip(xs, self.embs, self.emb_dropouts)
            ]
        if self.hparams.use_cat:
            content_annotations = torch.cat(content_annotations, dim = -1)
        else :
            content_annotations = sum(content_annotations)
        if self.pretrain_dim != 0:
            content_annotations = torch.cat([content_annotations, self.pretrain_emb_dropout(self.pretrain_emb(pre_words_idxs), batch_idxs)], dim  = 1)

        if extra_content_annotations is not None:
            if self.extra_content_dropout is not None:
                extra_content_annotations = self.extra_content_dropout(extra_content_annotations, batch_idxs)

            if self.hparams.use_cat:
                content_annotations = torch.cat(
                    [content_annotations, extra_content_annotations], dim=-1)
            else:
                content_annotations += extra_content_annotations

        timing_signal = []
        for seq_len in batch_idxs.seq_lens_np:
            this_seq_len = seq_len
            timing_signal.append(self.position_table[:this_seq_len,:])
            this_seq_len -= self.max_len
            while this_seq_len > 0:
                timing_signal.append(self.position_table[:this_seq_len,:])
                this_seq_len -= self.max_len

        timing_signal = torch.cat(timing_signal, dim=0)
        timing_signal = self.timing_dropout(timing_signal, batch_idxs)

        # Combine the content and timing signals
        if self.partitioned:
            annotations = torch.cat([content_annotations, timing_signal], 1)
        else:
            annotations = content_annotations + timing_signal

        #print(annotations.shape)
        annotations = self.layer_norm(self.dropout(annotations, batch_idxs))
        content_annotations = self.dropout(content_annotations, batch_idxs)

        return annotations, content_annotations, timing_signal, batch_idxs

#

class CharacterLSTM(nn.Module):
    def __init__(self, num_embeddings, d_embedding, d_out,
            char_dropout=0.0,
            normalize=False,
            **kwargs):
        super(CharacterLSTM, self).__init__()

        self.d_embedding = d_embedding
        self.d_out = d_out

        self.lstm = nn.LSTM(self.d_embedding, self.d_out // 2, num_layers=1, bidirectional=True)

        self.emb = nn.Embedding(num_embeddings, self.d_embedding, **kwargs)
        self.char_dropout = nn.Dropout(char_dropout)

        if normalize:
            print("This experiment: layer-normalizing after character LSTM")
            self.layer_norm = LayerNormalization(self.d_out, affine=False)
        else:
            self.layer_norm = lambda x: x

    def forward(self, chars_padded_np, word_lens_np, batch_idxs):
        # copy to ensure nonnegative stride for successful transfer to pytorch
        decreasing_idxs_np = np.argsort(word_lens_np)[::-1].copy()
        decreasing_idxs_torch = from_numpy(decreasing_idxs_np)
        decreasing_idxs_torch.requires_grad_(False)

        chars_padded = from_numpy(chars_padded_np[decreasing_idxs_np])
        chars_padded.requires_grad_(False)
        word_lens = from_numpy(word_lens_np[decreasing_idxs_np])

        inp_sorted = nn.utils.rnn.pack_padded_sequence(chars_padded, word_lens_np[decreasing_idxs_np], batch_first=True)
        inp_sorted_emb = nn.utils.rnn.PackedSequence(
            self.char_dropout(self.emb(inp_sorted.data)),
            inp_sorted.batch_sizes)
        _, (lstm_out, _) = self.lstm(inp_sorted_emb)

        lstm_out = torch.cat([lstm_out[0], lstm_out[1]], -1)

        # Undo sorting by decreasing word length
        res = torch.zeros_like(lstm_out)
        res.index_copy_(0, decreasing_idxs_torch, lstm_out)

        res = self.layer_norm(res)
        return res

def get_elmo_class():
    # Avoid a hard dependency by only importing Elmo if it's being used
    from allennlp.modules.elmo import Elmo

    class ModElmo(Elmo):
       def forward(self, inputs):
            """
            Unlike Elmo.forward, return vector representations for bos/eos tokens

            This modified version does not support extra tensor dimensions

            Parameters
            ----------
            inputs : ``torch.autograd.Variable``
                Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.

            Returns
            -------
            Dict with keys:
            ``'elmo_representations'``: ``List[torch.autograd.Variable]``
                A ``num_output_representations`` list of ELMo representations for the input sequence.
                Each representation is shape ``(batch_size, timesteps + 2, embedding_dim)``
            ``'mask'``:  ``torch.autograd.Variable``
                Shape ``(batch_size, timesteps + 2)`` long tensor with sequence mask.
            """
            # reshape the input if needed
            original_shape = inputs.size()
            timesteps, num_characters = original_shape[-2:]
            assert len(original_shape) == 3, "Only 3D tensors supported here"
            reshaped_inputs = inputs

            # run the biLM
            bilm_output = self._elmo_lstm(reshaped_inputs)
            layer_activations = bilm_output['activations']
            mask_with_bos_eos = bilm_output['mask']

            # compute the elmo representations
            representations = []
            for i in range(len(self._scalar_mixes)):
                scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
                representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
                # We don't remove bos/eos here!
                representations.append(self._dropout(representation_with_bos_eos))

            mask = mask_with_bos_eos
            elmo_representations = representations

            return {'elmo_representations': elmo_representations, 'mask': mask}
    return ModElmo

def get_xlnet(xlnet_model, xlnet_do_lower_case):
    # Avoid a hard dependency on BERT by only importing it if it's being used
    from transformers import (WEIGHTS_NAME, XLNetModel,
                                      XLMConfig, XLMForSequenceClassification,
                                      XLMTokenizer, XLNetConfig,
                                      XLNetForSequenceClassification,
                                      XLNetTokenizer)
    tokenizer = XLNetTokenizer.from_pretrained(xlnet_model, do_lower_case=xlnet_do_lower_case)
    xlnet = XLNetModel.from_pretrained(xlnet_model)

    return tokenizer, xlnet

def get_roberta(roberta_model, roberta_do_lower_case):
    # Avoid a hard dependency on BERT by only importing it if it's being used
    from transformers import (WEIGHTS_NAME, RobertaModel,
                                      RobertaConfig,
                                      RobertaForSequenceClassification,
                                      RobertaTokenizer)
    tokenizer = RobertaTokenizer.from_pretrained(roberta_model, do_lower_case=roberta_do_lower_case, add_special_tokens=True)
    roberta = RobertaModel.from_pretrained(roberta_model)

    return tokenizer, roberta


def get_bert(bert_model, bert_do_lower_case):
    # Avoid a hard dependency on BERT by only importing it if it's being used
    from pretrained_bert import BertTokenizer, BertModel
    if bert_model.endswith('.tar.gz'):
        tokenizer = BertTokenizer.from_pretrained(bert_model.replace('.tar.gz', '-vocab.txt'), do_lower_case=bert_do_lower_case)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
    bert = BertModel.from_pretrained(bert_model)
    return tokenizer, bert

# def get_bert(bert_model, bert_do_lower_case):
#     # Avoid a hard dependency on BERT by only importing it if it's being used
#     from pytorch_transformers import BertTokenizer, BertModel
#     if bert_model.endswith('.tar.gz'):
#         tokenizer = BertTokenizer.from_pretrained(bert_model.replace('.tar.gz', '-vocab.txt'), do_lower_case=bert_do_lower_case)
#     else:
#         tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
#     bert = BertModel.from_pretrained(bert_model)
#     return tokenizer, bert

#
class BiLinear(nn.Module):
    '''
    Bi-linear layer
    '''
    def __init__(self, left_features, right_features, out_features, bias=True):
        '''

        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        '''
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = nn.Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = nn.Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = nn.Parameter(torch.Tensor(self.out_features, self.left_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        '''

        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]

        Returns:

        '''
        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(-1, self.left_features)
        input_right = input_right.view(-1, self.right_features)

        # output [batch, out_features]
        output = nn.functional.bilinear(input_left, input_right, self.U, self.bias)
        output = output + nn.functional.linear(input_left, self.W_l, None) + nn.functional.linear(input_right, self.W_r, None)
        # convert back to [batch1, batch2, ..., out_features]
        return output
#
class BiAAttention(nn.Module):
    '''
    Bi-Affine attention layer.
    '''

    def __init__(self, hparams):
        super(BiAAttention, self).__init__()
        self.hparams = hparams

        self.dep_weight = nn.Parameter(torch_t.FloatTensor(hparams.d_biaffine + 1, hparams.d_biaffine + 1))
        nn.init.xavier_uniform_(self.dep_weight)

    def forward(self, input_d, input_e, input_s = None):

        score = torch.matmul(torch.cat(
            [input_d, torch_t.FloatTensor(input_d.size(0), 1).fill_(1).requires_grad_(False)],
            dim=1), self.dep_weight)
        score1 = torch.matmul(score, torch.transpose(torch.cat(
            [input_e, torch_t.FloatTensor(input_e.size(0), 1).fill_(1).requires_grad_(False)],
            dim=1), 0, 1))

        return score1

class Dep_score(nn.Module):
    def __init__(self, hparams, num_labels):
        super(Dep_score, self).__init__()

        self.dropout_out = nn.Dropout2d(p=0.33)
        self.hparams = hparams
        out_dim = hparams.d_biaffine#d_biaffine
        self.arc_h = nn.Linear(hparams.annotation_dim, hparams.d_biaffine)
        self.arc_c = nn.Linear(hparams.annotation_dim, hparams.d_biaffine)

        self.attention = BiAAttention(hparams)

        self.type_h = nn.Linear(hparams.annotation_dim, hparams.d_label_hidden)
        self.type_c = nn.Linear(hparams.annotation_dim, hparams.d_label_hidden)
        self.bilinear = BiLinear(hparams.d_label_hidden, hparams.d_label_hidden, num_labels)

    def forward(self, outputs, outpute):
        # output from rnn [batch, length, hidden_size]

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        outpute = self.dropout_out(outpute.transpose(1, 0)).transpose(1, 0)
        outputs = self.dropout_out(outputs.transpose(1, 0)).transpose(1, 0)

        # output size [batch, length, arc_space]
        arc_h = nn.functional.relu(self.arc_h(outputs))
        arc_c = nn.functional.relu(self.arc_c(outpute))

        # output size [batch, length, type_space]
        type_h = nn.functional.relu(self.type_h(outputs))
        type_c = nn.functional.relu(self.type_c(outpute))

        # apply dropout
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=0)
        type = torch.cat([type_h, type_c], dim=0)

        arc = self.dropout_out(arc.transpose(1, 0)).transpose(1, 0)
        arc_h, arc_c = arc.chunk(2, 0)

        type = self.dropout_out(type.transpose(1, 0)).transpose(1, 0)
        type_h, type_c = type.chunk(2, 0)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        out_arc = self.attention(arc_h, arc_c)
        out_type = self.bilinear(type_h, type_c)

        return out_arc, out_type

class LabelAttention(nn.Module):
    """
    Single-head Attention layer for label-specific representations
    """

    def __init__(self, hparams, d_model, d_k, d_v, d_l, d_proj, use_resdrop=True, q_as_matrix=False, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        super(LabelAttention, self).__init__()
        self.hparams = hparams
        self.d_k = d_k
        self.d_v = d_v
        self.d_l = d_l # Number of Labels
        self.d_model = d_model # Model Dimensionality
        self.d_proj = d_proj # Projection dimension of each label output
        self.use_resdrop = use_resdrop # Using Residual Dropout?
        self.q_as_matrix = q_as_matrix # Using a Matrix of Q to be multiplied with input instead of learned q vectors
        self.combine_as_self = hparams.lal_combine_as_self # Using the Combination Method of Self-Attention

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            if self.q_as_matrix:
                self.w_qs1 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
            else:
                self.w_qs1 = nn.Parameter(torch_t.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
            self.w_ks1 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
            self.w_vs1 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_content, d_v // 2), requires_grad=True)

            if self.q_as_matrix:
                self.w_qs2 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_positional, d_k // 2), requires_grad=True)
            else:
                self.w_qs2 = nn.Parameter(torch_t.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
            self.w_ks2 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_positional, d_k // 2), requires_grad=True)
            self.w_vs2 = nn.Parameter(torch_t.FloatTensor(self.d_l, self.d_positional, d_v // 2), requires_grad=True)

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            if self.q_as_matrix:
                self.w_qs = nn.Parameter(torch_t.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
            else:
                self.w_qs = nn.Parameter(torch_t.FloatTensor(self.d_l, d_k), requires_grad=True)
            self.w_ks = nn.Parameter(torch_t.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
            self.w_vs = nn.Parameter(torch_t.FloatTensor(self.d_l, d_model, d_v), requires_grad=True)

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        if self.combine_as_self:
            self.layer_norm = LayerNormalization(d_model)
        else:
            self.layer_norm = LayerNormalization(self.d_proj)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            if self.combine_as_self:
                self.proj = nn.Linear(self.d_l * d_v, d_model, bias=False)
            else:
                self.proj = nn.Linear(d_v, d_model, bias=False) # input dimension does not match, should be d_l * d_v
        else:
            if self.combine_as_self:
                self.proj1 = nn.Linear(self.d_l*(d_v//2), self.d_content, bias=False)
                self.proj2 = nn.Linear(self.d_l*(d_v//2), self.d_positional, bias=False)
            else:
                self.proj1 = nn.Linear(d_v//2, self.d_content, bias=False)
                self.proj2 = nn.Linear(d_v//2, self.d_positional, bias=False)
        if not self.combine_as_self:
            self.reduce_proj = nn.Linear(d_model, self.d_proj, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, k_inp=None):
        len_inp = inp.size(0)
        v_inp_repeated = inp.repeat(self.d_l, 1).view(self.d_l, -1, inp.size(-1)) # d_l x len_inp x d_model
        if k_inp is None:
            k_inp_repeated = v_inp_repeated
        else:
            k_inp_repeated = k_inp.repeat(self.d_l, 1).view(self.d_l, -1, k_inp.size(-1)) # d_l x len_inp x d_model

        if not self.partitioned:
            if self.q_as_matrix:
                q_s = torch.bmm(k_inp_repeated, self.w_qs) # d_l x len_inp x d_k
            else:
                q_s = self.w_qs.unsqueeze(1) # d_l x 1 x d_k
            k_s = torch.bmm(k_inp_repeated, self.w_ks) # d_l x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs) # d_l x len_inp x d_v
        else:
            if self.q_as_matrix:
                q_s = torch.cat([
                    torch.bmm(k_inp_repeated[:,:,:self.d_content], self.w_qs1),
                    torch.bmm(k_inp_repeated[:,:,self.d_content:], self.w_qs2),
                    ], -1)
            else:
                q_s = torch.cat([
                    self.w_qs1.unsqueeze(1),
                    self.w_qs2.unsqueeze(1),
                    ], -1)
            k_s = torch.cat([
                torch.bmm(k_inp_repeated[:,:,:self.d_content], self.w_ks1),
                torch.bmm(k_inp_repeated[:,:,self.d_content:], self.w_ks2),
                ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:,:,self.d_content:], self.w_vs2),
                ], -1)
        return q_s, k_s, v_s
    
    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * mb_size) x len_padded x d
        # (along with masks for the attention and output)
        n_head = self.d_l
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        if self.q_as_matrix:
            q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        else:
            q_padded = q_s.repeat(mb_size, 1, 1) # (d_l * mb_size) x 1 x d_k
        k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=DTYPE)

        for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            if self.q_as_matrix:
                q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
            k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
            v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
            invalid_mask[i, :end-start].fill_(False)

        if self.q_as_matrix:
            q_padded = q_padded.view(-1, len_padded, d_k)
            attn_mask = invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1)
        else:
            attn_mask = invalid_mask.unsqueeze(1).repeat(n_head, 1, 1)
        
        output_mask = (~invalid_mask).repeat(n_head, 1)

        return(
            q_padded,
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            attn_mask,
            output_mask,
            )

    def combine_v(self, outputs):
        # Combine attention information from the different labels
        d_l = self.d_l
        outputs = outputs.view(d_l, -1, self.d_v) # d_l x len_inp x d_v

        if not self.partitioned:
            # Switch from d_l x len_inp x d_v to len_inp x d_l x d_v
            if self.combine_as_self:
                outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, d_l * self.d_v)
            else:
                outputs = torch.transpose(outputs, 0, 1)#.contiguous() #.view(-1, d_l * self.d_v)
            # Project back to residual size
            outputs = self.proj(outputs) # Becomes len_inp x d_l x d_model
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:,:,:d_v1]
            outputs2 = outputs[:,:,d_v1:]
            if self.combine_as_self:
                outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, d_l * d_v1)
                outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, d_l * d_v1)
            else:
                outputs1 = torch.transpose(outputs1, 0, 1)#.contiguous() #.view(-1, d_l * d_v1)
                outputs2 = torch.transpose(outputs2, 0, 1)#.contiguous() #.view(-1, d_l * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
                ], -1)#.contiguous()

        return outputs

    def forward(self, inp, batch_idxs, k_inp=None):
        residual = inp # len_inp x d_model
        len_inp = inp.size(0)

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, k_inp=k_inp)
        # d_l x len_inp x d_k
        # q_s is d_l x 1 x d_k

        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)
        # q_padded, k_padded, v_padded: (d_l * batch_size) x max_len x d_kv
        # q_s is (d_l * batch_size) x 1 x d_kv

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask,
            )
        # outputs_padded: (d_l * batch_size) x max_len x d_kv
        # in LAL: (d_l * batch_size) x 1 x d_kv
        # on the best model, this is one value vector per label that is repeated max_len times
        if not self.q_as_matrix:
            outputs_padded = outputs_padded.repeat(1,output_mask.size(-1),1)
        outputs = outputs_padded[output_mask]
        # outputs: (d_l * len_inp) x d_kv or LAL: (d_l * len_inp) x d_kv
        # output_mask: (d_l * batch_size) x max_len
        torch.cuda.empty_cache()
        outputs = self.combine_v(outputs)
        # outputs: len_inp x d_l x d_model, whereas a normal self-attention layer gets len_inp x d_model
        if self.use_resdrop:
            if self.combine_as_self:
                outputs = self.residual_dropout(outputs, batch_idxs)
            else:
                outputs = torch.cat([self.residual_dropout(outputs[:,i,:], batch_idxs).unsqueeze(1) for i in range(self.d_l)], 1)
        if self.combine_as_self:
            outputs = self.layer_norm(outputs + inp)
        else:
            for l in range(self.d_l):
                outputs[:, l, :] = outputs[:, l, :] + inp
            
            outputs = self.reduce_proj(outputs) # len_inp x d_l x d_proj
            outputs = self.layer_norm(outputs) # len_inp x d_l x d_proj
            outputs = outputs.view(len_inp, -1).contiguous() # len_inp x (d_l * d_proj)
        
        return outputs, attns_padded

class Encoder(nn.Module):
    def __init__(self, hparams, embedding,
                    num_layers=1, num_heads=2, d_kv = 32, d_ff=1024, d_l=112,
                    d_positional=None,
                    num_layers_position_only=0,
                    relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1,
                    use_lal=True,
                    lal_d_kv=128,
                    lal_d_proj=128,
                    lal_resdrop=True,
                    lal_pwff=True,
                    lal_q_as_matrix=False,
                    lal_partitioned=True):
        super().__init__()
        self.embedding_container = [embedding]
        d_model = embedding.d_embedding
        self.hparams = hparams

        d_k = d_v = d_kv

        self.stacks = []

        for i in range(hparams.num_layers):
            attn = MultiHeadAttention(hparams, num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout,
                                      attention_dropout=attention_dropout, d_positional=d_positional)
            if d_positional is None:
                ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout,
                                             residual_dropout=residual_dropout)
            else:
                ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout,
                                                        residual_dropout=residual_dropout)

            self.add_module(f"attn_{i}", attn)
            self.add_module(f"ff_{i}", ff)

            self.stacks.append((attn, ff))

        if use_lal:
            lal_d_positional = d_positional if lal_partitioned else None
            attn = LabelAttention(hparams, d_model, lal_d_kv, lal_d_kv, d_l, lal_d_proj, use_resdrop=lal_resdrop, q_as_matrix=lal_q_as_matrix,
                                  residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=lal_d_positional)
            ff_dim = lal_d_proj * d_l
            if hparams.lal_combine_as_self:
                ff_dim = d_model
            if lal_pwff:
                if d_positional is None or not lal_partitioned:
                    ff = PositionwiseFeedForward(ff_dim, d_ff, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
                else:
                    ff = PartitionedPositionwiseFeedForward(ff_dim, d_ff, d_positional, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
            else:
                ff = None

            self.add_module(f"attn_{num_layers}", attn)
            self.add_module(f"ff_{num_layers}", ff)
            self.stacks.append((attn, ff))

        self.num_layers_position_only = num_layers_position_only
        if self.num_layers_position_only > 0:
            assert d_positional is None, "num_layers_position_only and partitioned are incompatible"

    def forward(self, xs, pre_words_idxs, batch_idxs, extra_content_annotations=None):
        emb = self.embedding_container[0]
        res, res_c, timing_signal, batch_idxs = emb(xs, pre_words_idxs, batch_idxs, extra_content_annotations=extra_content_annotations)

        for i, (attn, ff) in enumerate(self.stacks):
            res, current_attns = attn(res, batch_idxs)
            if ff is not None:
                res = ff(res, batch_idxs)

        return res, current_attns #batch_idxs

class ChartParser(nn.Module):
    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            type_vocab,
            hparams,
            exp_name=None
    ):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec['hparams'] = hparams.to_dict()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.type_vocab = type_vocab

        self.hparams = hparams
        self.d_model = hparams.d_model
        self.partitioned = hparams.partitioned
        self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
        self.d_positional = (hparams.d_model // 2) if self.partitioned else None

        self.use_lal = hparams.use_lal
        if self.use_lal:
            self.lal_d_kv = hparams.lal_d_kv
            self.lal_d_proj = hparams.lal_d_proj
            self.lal_resdrop = hparams.lal_resdrop
            self.lal_pwff = hparams.lal_pwff
            self.lal_q_as_matrix = hparams.lal_q_as_matrix
            self.lal_partitioned = hparams.lal_partitioned
            self.lal_combine_as_self = hparams.lal_combine_as_self

        self.contributions = False

        num_embeddings_map = {
            'tags': tag_vocab.size,
            'words': word_vocab.size,
            'chars': char_vocab.size,
        }
        emb_dropouts_map = {
            'tags': hparams.tag_emb_dropout,
            'words': hparams.word_emb_dropout,
        }

        self.emb_types = []
        if hparams.use_tags:
            self.emb_types.append('tags')
        if hparams.use_words:
            self.emb_types.append('words')

        self.use_tags = hparams.use_tags

        self.morpho_emb_dropout = None

        self.char_encoder = None
        self.elmo = None
        self.bert = None
        self.xlnet = None
        self.pad_left = hparams.pad_left
        self.roberta = None
        ex_dim = self.d_content
        if self.hparams.use_cat:
            cun = 0
            if hparams.use_words or hparams.use_tags:
                ex_dim = ex_dim // 2 #word dim = self.d_content/2
            if hparams.use_chars_lstm:
                cun = cun+1
            if hparams.use_elmo or hparams.use_bert or hparams.use_xlnet:
                cun = cun + 1
            if cun > 0 :
                ex_dim = ex_dim // cun
        if hparams.use_chars_lstm:
            self.char_encoder = CharacterLSTM(
                num_embeddings_map['chars'],
                hparams.d_char_emb,
                ex_dim,
                char_dropout=hparams.char_lstm_input_dropout,
            )
        if hparams.use_elmo:
            self.elmo = get_elmo_class()(
                options_file="data/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                weight_file="data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                num_output_representations=1,
                requires_grad=False,
                do_layer_norm=False,
                dropout=hparams.elmo_dropout,
                )
            d_elmo_annotations = 1024

            # Don't train gamma parameter for ELMo - the projection can do any
            # necessary scaling
            self.elmo.scalar_mix_0.gamma.requires_grad = False

            # Reshapes the embeddings to match the model dimension, and making
            # the projection trainable appears to improve parsing accuracy
            self.project_elmo = nn.Linear(d_elmo_annotations, ex_dim, bias=False)

        if hparams.use_xlnet:

            self.xlnet_tokenizer, self.xlnet = get_xlnet(hparams.xlnet_model, hparams.xlnet_do_lower_case)
            if hparams.bert_transliterate:
                from transliterate import TRANSLITERATIONS
                self.bert_transliterate = TRANSLITERATIONS[hparams.bert_transliterate]
            else:
                self.bert_transliterate = None

            d_xlnet_annotations = self.xlnet.d_model
            self.xlnet_max_len = 768

            self.project_xlnet = nn.Linear(d_xlnet_annotations, ex_dim, bias=False)

        if hparams.use_bert or hparams.use_bert_only:
            self.bert_tokenizer, self.bert = get_bert(hparams.bert_model, hparams.bert_do_lower_case)
            if hparams.bert_transliterate:
                from transliterate import TRANSLITERATIONS
                self.bert_transliterate = TRANSLITERATIONS[hparams.bert_transliterate]
            else:
                self.bert_transliterate = None

            d_bert_annotations = self.bert.pooler.dense.in_features
            self.bert_max_len = self.bert.embeddings.position_embeddings.num_embeddings

            if hparams.use_bert_only:
                self.project_bert = nn.Linear(d_bert_annotations, hparams.d_model, bias=False)
            else:
                self.project_bert = nn.Linear(d_bert_annotations, ex_dim, bias=False)
        
        if hparams.use_roberta:
            self.roberta_tokenizer, self.roberta = get_roberta(hparams.roberta_model, hparams.roberta_do_lower_case)
            if hparams.bert_transliterate:
                from transliterate import TRANSLITERATIONS
                self.bert_transliterate = TRANSLITERATIONS[hparams.bert_transliterate]
            else:
                self.bert_transliterate = None

            d_roberta_annotations = self.roberta.pooler.dense.in_features
            self.roberta_max_len = self.roberta.embeddings.position_embeddings.num_embeddings

            self.project_roberta = nn.Linear(d_roberta_annotations, self.d_content, bias=False)

        if not hparams.use_bert_only:

            if hparams.embedding_type != 'random' and hparams.use_words:
                embedd_dict, embedd_dim = utils.load_embedding_dict(hparams.embedding_type, hparams.embedding_path)
                scale = np.sqrt(3.0 / embedd_dim)
                table = np.zeros([word_vocab.size, embedd_dim], dtype=np.float32)
                oov = 0
                for index, word in enumerate(word_vocab.indices):
                    if word in embedd_dict:
                        embedding = embedd_dict[word]
                    else:
                        embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                        oov += 1
                    table[index, :embedd_dim] = embedding
                print('oov: %d' % oov)
                word_table_np = table
                # self.project_pretrain = nn.Linear(embedd_dim, self.d_content, bias=False)
            else:
                word_table_np = None

            self.embedding = MultiLevelEmbedding(
                [num_embeddings_map[emb_type] for emb_type in self.emb_types],
                hparams.d_model,
                hparams=hparams,
                d_positional=self.d_positional,
                dropout=hparams.embedding_dropout,
                timing_dropout=hparams.timing_dropout,
                emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
                extra_content_dropout=self.morpho_emb_dropout,
                max_len=hparams.sentence_max_len,
                word_table_np=word_table_np,
            )

            self.encoder = Encoder(
                hparams,
                self.embedding,
                num_layers=hparams.num_layers,
                num_heads=hparams.num_heads,
                d_kv=hparams.d_kv,
                d_ff=hparams.d_ff,
                d_positional=self.d_positional,
                relu_dropout=hparams.relu_dropout,
                residual_dropout=hparams.residual_dropout,
                attention_dropout=hparams.attention_dropout,
                use_lal=hparams.use_lal,
                lal_d_kv=hparams.lal_d_kv,
                lal_d_proj=hparams.lal_d_proj,
                lal_resdrop=hparams.lal_resdrop,
                lal_pwff=hparams.lal_pwff,
                lal_q_as_matrix=hparams.lal_q_as_matrix,
                lal_partitioned=hparams.lal_partitioned,
            )
        else:
            self.embedding = None
            self.encoder = None

        annotation_dim = ((label_vocab.size - 2) * self.lal_d_proj) if (self.use_lal and not self.lal_combine_as_self) else hparams.d_model
        hparams.annotation_dim = annotation_dim

        self.f_label = nn.Sequential(
            nn.Linear(annotation_dim, hparams.d_label_hidden),
            LayerNormalization(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, label_vocab.size - 1 ),
            )
        self.dep_score = Dep_score(hparams, type_vocab.size)
        self.loss_func = torch.nn.CrossEntropyLoss(size_average=False)
        self.loss_funt = torch.nn.CrossEntropyLoss(size_average=False)

        if not hparams.use_tags and hasattr(hparams, 'd_tag_hidden'):
            self.f_tag = nn.Sequential(
                nn.Linear(annotation_dim, hparams.d_tag_hidden),
                LayerNormalization(hparams.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_tag_hidden, tag_vocab.size),
                )
            self.tag_loss_scale = hparams.tag_loss_scale
        else:
            self.f_tag = None

        if use_cuda:
            self.cuda()
            
    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model):
        spec = spec.copy()
        hparams = spec['hparams']
        if 'sentence_max_len' not in hparams:
            hparams['sentence_max_len'] = 300
        if 'use_elmo' not in hparams:
            hparams['use_elmo'] = False
        if 'elmo_dropout' not in hparams:
            hparams['elmo_dropout'] = 0.5
        if 'const_lada' not in hparams:
            hparams['const_lada'] = 0.5
        if 'use_bert' not in hparams:
            hparams['use_bert'] = False
        if 'use_bert_only' not in hparams:
            hparams['use_bert_only'] = False
        if 'bert_model' not in hparams:
            hparams['bert_model'] = "bert-large-uncased"
        if 'bert_do_lower_case' not in hparams:
            hparams['bert_do_lower_case'] = False
        if 'bert_transliterate' not in hparams:
            hparams['bert_transliterate'] = ""

        if 'use_xlnet' not in hparams:
            hparams['use_xlnet'] = False
        if 'use_roberta' not in hparams:
            hparams['use_roberta'] = False
        if 'xlnet_model' not in hparams:
            hparams['xlnet_model'] = "xlnet-large-cased"
        if 'xlnet_do_lower_case' not in hparams:
            hparams['xlnet_do_lower_case'] = False
        if 'pad_left' not in hparams:
            hparams['pad_left'] = False

        for param in 'use_lal lal_resdrop lal_pwff lal_partitioned lal_q_as_matrix lal_combine_as_self'.split(' '):
            if param not in hparams:
                hparams[param] = False
        for param in 'lal_d_kv lal_d_proj'.split(' '):
            if param not in hparams:
                hparams[param] = 64

        spec['hparams'] = makehp.HParams(**hparams)
        res = cls(**spec)
        if use_cuda:
            res.cpu()
        if not hparams['use_elmo']:
            res.load_state_dict(model)
        else:
            state = {k: v for k,v in res.state_dict().items() if k not in model}
            state.update(model)
            res.load_state_dict(state)
        if use_cuda:
            res.cuda()
        return res

    def split_batch(self, sentences, golds, subbatch_max_tokens=3000):
        lens = [len(sentence) + 2 for sentence in sentences]

        lens = np.asarray(lens, dtype=int)
        lens_argsort = np.argsort(lens).tolist()

        num_subbatches = 0
        subbatch_size = 1
        while lens_argsort:
            if (subbatch_size == len(lens_argsort)) or (subbatch_size * lens[lens_argsort[subbatch_size]] > subbatch_max_tokens):
                yield [sentences[i] for i in lens_argsort[:subbatch_size]], [golds[i] for i in lens_argsort[:subbatch_size]]
                lens_argsort = lens_argsort[subbatch_size:]
                num_subbatches += 1
                subbatch_size = 1
            else:
                subbatch_size += 1

    def parse_batch(self, sentences, golds=None):
        is_train = golds is not None
        self.train(is_train)
        torch.set_grad_enabled(is_train)
        self.current_attns = None

        if golds is None:
            golds = [None] * len(sentences)

        packed_len = sum([(len(sentence) + 2) for sentence in sentences])

        i = 0
        tag_idxs = np.zeros(packed_len, dtype=int)
        word_idxs = np.zeros(packed_len, dtype=int)
        batch_idxs = np.zeros(packed_len, dtype=int)
        for snum, sentence in enumerate(sentences):
            #print(sentence)
            for (tag, word) in [(START, START)] + sentence + [(STOP, STOP)]:
                tag_idxs[i] = 0 if (not self.use_tags and self.f_tag is None) else self.tag_vocab.index_or_unk(tag, TAG_UNK)
                if word not in (START, STOP):
                    count = self.word_vocab.count(word)
                    if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                        word = UNK
                word_idxs[i] = self.word_vocab.index(word)
                batch_idxs[i] = snum
                i += 1
        assert i == packed_len

        batch_idxs = BatchIndices(batch_idxs)

        emb_idxs_map = {
            'tags': tag_idxs,
            'words': word_idxs,
        }
        emb_idxs = [
            from_numpy(emb_idxs_map[emb_type]).requires_grad_(False)
            for emb_type in self.emb_types
        ]
        pre_words_idxs = from_numpy(word_idxs).requires_grad_(False)
        
        if is_train and self.f_tag is not None:
            gold_tag_idxs = from_numpy(emb_idxs_map['tags'])

        extra_content_annotations_list = []
        extra_content_annotations = None
        if self.char_encoder is not None:
            assert isinstance(self.char_encoder, CharacterLSTM)
            max_word_len = max([max([len(word) for tag, word in sentence]) for sentence in sentences])
            # Add 2 for start/stop tokens
            max_word_len = max(max_word_len, 3) + 2
            char_idxs_encoder = np.zeros((packed_len, max_word_len), dtype=int)
            word_lens_encoder = np.zeros(packed_len, dtype=int)
            i = 0
            for snum, sentence in enumerate(sentences):
                for wordnum, (tag, word) in enumerate([(START, START)] + sentence + [(STOP, STOP)]):
                    j = 0
                    char_idxs_encoder[i, j] = self.char_vocab.index(CHAR_START_WORD)
                    j += 1
                    if word in (START, STOP):
                        char_idxs_encoder[i, j:j + 3] = self.char_vocab.index(
                            CHAR_START_SENTENCE if (word == START) else CHAR_STOP_SENTENCE
                        )
                        j += 3
                    else:
                        for char in word:
                            char_idxs_encoder[i, j] = self.char_vocab.index_or_unk(char, CHAR_UNK)
                            j += 1
                    char_idxs_encoder[i, j] = self.char_vocab.index(CHAR_STOP_WORD)
                    word_lens_encoder[i] = j + 1
                    i += 1
            assert i == packed_len

            extra_content_annotations_list.append(self.char_encoder(char_idxs_encoder, word_lens_encoder, batch_idxs))
        if self.elmo is not None:
            # See https://github.com/allenai/allennlp/blob/c3c3549887a6b1fb0bc8abf77bc820a3ab97f788/allennlp/data/token_indexers/elmo_indexer.py#L61
            # ELMO_START_SENTENCE = 256
            # ELMO_STOP_SENTENCE = 257
            ELMO_START_WORD = 258
            ELMO_STOP_WORD = 259
            ELMO_CHAR_PAD = 260

            # Sentence start/stop tokens are added inside the ELMo module
            max_sentence_len = max([(len(sentence)) for sentence in sentences])
            max_word_len = 50
            char_idxs_encoder = np.zeros((len(sentences), max_sentence_len, max_word_len), dtype=int)

            for snum, sentence in enumerate(sentences):
                for wordnum, (tag, word) in enumerate(sentence):
                    char_idxs_encoder[snum, wordnum, :] = ELMO_CHAR_PAD

                    j = 0
                    char_idxs_encoder[snum, wordnum, j] = ELMO_START_WORD
                    j += 1
                    assert word not in (START, STOP)
                    for char_id in word.encode('utf-8', 'ignore')[:(max_word_len-2)]:
                        char_idxs_encoder[snum, wordnum, j] = char_id
                        j += 1
                    char_idxs_encoder[snum, wordnum, j] = ELMO_STOP_WORD

                    # +1 for masking (everything that stays 0 is past the end of the sentence)
                    char_idxs_encoder[snum, wordnum, :] += 1

            char_idxs_encoder = from_numpy(char_idxs_encoder).requires_grad_(False)

            elmo_out = self.elmo.forward(char_idxs_encoder)
            elmo_rep0 = elmo_out['elmo_representations'][0]
            elmo_mask = elmo_out['mask']

            elmo_annotations_packed = elmo_rep0[elmo_mask.byte()].view(packed_len, -1)

            # Apply projection to match dimensionality
            extra_content_annotations_list.append(self.project_elmo(elmo_annotations_packed))

        if self.bert is not None:
            all_input_ids = np.zeros((len(sentences), self.bert_max_len), dtype=int)
            all_input_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)
            all_word_start_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)
            all_word_end_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)

            subword_max_len = 0
            for snum, sentence in enumerate(sentences):
                tokens = []
                word_start_mask = []
                word_end_mask = []

                tokens.append("[CLS]")
                word_start_mask.append(1)
                word_end_mask.append(1)

                if self.bert_transliterate is None:
                    cleaned_words = []
                    for _, word in sentence:
                        word = BERT_TOKEN_MAPPING.get(word, word)
                        if word == "n't" and cleaned_words:
                            cleaned_words[-1] = cleaned_words[-1] + "n"
                            word = "'t"
                        cleaned_words.append(word)
                else:
                    # When transliterating, assume that the token mapping is
                    # taken care of elsewhere
                    cleaned_words = [self.bert_transliterate(word) for _, word in sentence]

                for word in cleaned_words:
                    word_tokens = self.bert_tokenizer.tokenize(word)
                    for _ in range(len(word_tokens)):
                        word_start_mask.append(0)
                        word_end_mask.append(0)
                    word_start_mask[len(tokens)] = 1
                    word_end_mask[-1] = 1
                    tokens.extend(word_tokens)
                tokens.append("[SEP]")
                word_start_mask.append(1)
                word_end_mask.append(1)

                input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                subword_max_len = max(subword_max_len, len(input_ids))

                all_input_ids[snum, :len(input_ids)] = input_ids
                all_input_mask[snum, :len(input_mask)] = input_mask
                all_word_start_mask[snum, :len(word_start_mask)] = word_start_mask
                all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

            all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :subword_max_len]))
            all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :subword_max_len]))
            all_word_start_mask = from_numpy(np.ascontiguousarray(all_word_start_mask[:, :subword_max_len]))
            all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, :subword_max_len]))
            all_encoder_layers, _ = self.bert(all_input_ids, attention_mask=all_input_mask)
            del _
            features = all_encoder_layers[-1]

            if self.encoder is not None:
                features_packed = features.masked_select(all_word_end_mask.to(DTYPE).unsqueeze(-1)).reshape(-1,features.shape[-1])

                # For now, just project the features from the last word piece in each word
                extra_content_annotations = self.project_bert(features_packed)

        if self.xlnet is not None:
            #(XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            all_input_ids = np.zeros((len(sentences), self.xlnet_max_len), dtype=int)
            all_input_mask = np.zeros((len(sentences), self.xlnet_max_len), dtype=int)
            all_word_start_mask = np.zeros((len(sentences), self.xlnet_max_len), dtype=int)
            all_word_end_mask = np.zeros((len(sentences), self.xlnet_max_len), dtype=int)

            subword_max_len = 0
            for snum, sentence in enumerate(sentences):
                tokens = []
                word_start_mask = []
                word_end_mask = []

                # tokens.append("[CLS]")
                # word_start_mask.append(1)
                # word_end_mask.append(1)
                if not self.hparams.pad_left:
                    tokens.append(self.xlnet_tokenizer.cls_token)
                    word_start_mask.append(1)
                    word_end_mask.append(1)

                if self.bert_transliterate is None:
                    cleaned_words = []
                    for _, word in sentence:
                        word = BERT_TOKEN_MAPPING.get(word, word)
                        if word == "n't" and cleaned_words:
                            cleaned_words[-1] = cleaned_words[-1] + "n"
                            word = "'t"
                        cleaned_words.append(word)
                else:
                    # When transliterating, assume that the token mapping is
                    # taken care of elsewhere
                    cleaned_words = [self.bert_transliterate(word) for _, word in sentence]

                for word in cleaned_words:
                    word_tokens = self.xlnet_tokenizer.tokenize(word)
                    if len(word_tokens) ==0:
                        word_tokens = [self.xlnet_tokenizer.unk_token]
                    for _ in range(len(word_tokens)):
                        word_start_mask.append(0)
                        word_end_mask.append(0)
                    word_start_mask[len(tokens)] = 1
                    word_end_mask[-1] = 1
                    tokens.extend(word_tokens)
                tokens.append(self.xlnet_tokenizer.sep_token)
                word_start_mask.append(1)
                word_end_mask.append(1)

                if self.hparams.pad_left:
                    tokens.append(self.xlnet_tokenizer.cls_token)
                    word_start_mask.append(1)
                    word_end_mask.append(1)

                input_ids = self.xlnet_tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                subword_max_len = max(subword_max_len, len(input_ids))

                if self.hparams.pad_left:
                    all_input_ids[snum, self.xlnet_max_len - len(input_ids):] = input_ids
                    all_input_mask[snum, self.xlnet_max_len - len(input_mask):] = input_mask
                    all_word_start_mask[snum, self.xlnet_max_len - len(word_start_mask):] = word_start_mask
                    all_word_end_mask[snum, self.xlnet_max_len - len(word_end_mask):] = word_end_mask
                else:
                    all_input_ids[snum, :len(input_ids)] = input_ids
                    all_input_mask[snum, :len(input_mask)] = input_mask
                    all_word_start_mask[snum, :len(word_start_mask)] = word_start_mask
                    all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

            if self.hparams.pad_left:
                all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, self.xlnet_max_len - subword_max_len:]))
                all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, self.xlnet_max_len - subword_max_len:]))
                all_word_start_mask = from_numpy(np.ascontiguousarray(all_word_start_mask[:, self.xlnet_max_len - subword_max_len:]))
                all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, self.xlnet_max_len - subword_max_len:]))
            else:
                all_input_ids = from_numpy(
                    np.ascontiguousarray(all_input_ids[:, :subword_max_len]))
                all_input_mask = from_numpy(
                    np.ascontiguousarray(all_input_mask[:, :subword_max_len]))
                all_word_start_mask = from_numpy(
                    np.ascontiguousarray(all_word_start_mask[:, :subword_max_len]))
                all_word_end_mask = from_numpy(
                    np.ascontiguousarray(all_word_end_mask[:, :subword_max_len]))

            transformer_outputs = self.xlnet(all_input_ids, attention_mask=all_input_mask)
            # features = all_encoder_layers[-1]
            features = transformer_outputs[0]

            features_packed = features.masked_select(all_word_end_mask.to(DTYPE).unsqueeze(-1)).reshape(-1,
                                                                                                              features.shape[
                                                                                                                  -1])

            # For now, just project the features from the last word piece in each word
            extra_content_annotations = self.project_xlnet(features_packed)

        if self.encoder is not None:

            if len(extra_content_annotations_list) > 1 :
                if self.hparams.use_cat:
                    extra_content_annotations = torch.cat(extra_content_annotations_list, dim = -1)
                else:
                    extra_content_annotations = sum(extra_content_annotations_list)
            elif len(extra_content_annotations_list) == 1:
                extra_content_annotations = extra_content_annotations_list[0]

            annotations, self.current_attns = self.encoder(emb_idxs, pre_words_idxs, batch_idxs, extra_content_annotations=extra_content_annotations)

            if self.partitioned and not self.use_lal:
                annotations = torch.cat([
                    annotations[:, 0::2],
                    annotations[:, 1::2],
                ], 1)

            if self.use_lal and not self.lal_combine_as_self:
                half_dim = self.lal_d_proj//2
                d_l = self.label_vocab.size - 1
                fencepost_annotations = torch.cat(
                    [annotations[:-1, (i*self.lal_d_proj):(i*self.lal_d_proj + half_dim)] for i in range(d_l)] 
                    + [annotations[1:, (i*self.lal_d_proj + half_dim):((i+1)*self.lal_d_proj)] for i in range(d_l)], 1)
            else:
                fencepost_annotations = torch.cat([
                    annotations[:-1, :self.d_model//2],
                    annotations[1:, self.d_model//2:],
                    ], 1)

            fencepost_annotations_start = fencepost_annotations
            fencepost_annotations_end = fencepost_annotations

        else:
            assert self.bert is not None
            features = self.project_bert(features)
            fencepost_annotations_start = features.masked_select(all_word_start_mask.to(DTYPE).unsqueeze(-1)).reshape(-1, features.shape[-1])
            fencepost_annotations_end = features.masked_select(all_word_end_mask.to(DTYPE).unsqueeze(-1)).reshape(-1, features.shape[-1])
"""
        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        if not is_train:
            trees = []
            scores = []
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                tree, score = self.parse_from_annotations(fencepost_annotations_start[start:end,:],
                                                                        fencepost_annotations_end[start:end,:], sentences[i], i)
                trees.append(tree)
                scores.append(score)

            return trees, scores

        pis = []
        pjs = []
        plabels = []
        paugment_total = 0.0
        cun = 0
        num_p = 0
        gis = []
        gjs = []
        glabels = []
        with torch.no_grad():
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):

                p_i, p_j, p_label, p_augment, g_i, g_j, g_label \
                    = self.parse_from_annotations(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:], sentences[i], i, gold=golds[i])

                paugment_total += p_augment
                num_p += p_i.shape[0]
                pis.append(p_i + start)
                pjs.append(p_j + start)
                gis.append(g_i + start)
                gjs.append(g_j + start)
                plabels.append(p_label)
                glabels.append(g_label)

        cells_i = from_numpy(np.concatenate(pis + gis))
        cells_j = from_numpy(np.concatenate(pjs + gjs))
        cells_label = from_numpy(np.concatenate(plabels + glabels))

        cells_label_scores = self.f_label(fencepost_annotations_end[cells_j] - fencepost_annotations_start[cells_i])
        cells_label_scores = torch.cat([
            cells_label_scores.new_zeros((cells_label_scores.size(0), 1)),
            cells_label_scores
        ], 1)
        cells_label_scores = torch.gather(cells_label_scores, 1, cells_label[:, None])
        loss = cells_label_scores[:num_p].sum() - cells_label_scores[num_p:].sum() + paugment_total

        cun = 0
        for snum, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):

            #[start,....,end-1]->[<root>,1, 2,...,n]
            leng = end - start
            arc_score, type_score = self.dep_score(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:])
            #arc_gather = gfather[cun] - start
            arc_gather = [leaf.father for leaf in golds[snum].leaves()]
            type_gather = [self.type_vocab.index(leaf.type) for leaf in golds[snum].leaves()]
            cun += 1
            assert len(arc_gather) == leng - 1
            arc_score = torch.transpose(arc_score,0, 1)
            loss = loss + 0.5 * self.loss_func(arc_score[1:, :], from_numpy(np.array(arc_gather)).requires_grad_(False)) \
                   + 0.5 * self.loss_funt(type_score[1:, :],from_numpy(np.array(type_gather)).requires_grad_(False))

        return None, loss

    def label_scores_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end):

        span_features = (torch.unsqueeze(fencepost_annotations_end, 0)
                         - torch.unsqueeze(fencepost_annotations_start, 1))
        
        if self.contributions and self.use_lal:
            contributions = np.zeros((span_features.shape[0], span_features.shape[1], span_features.shape[2]//self.lal_d_proj))
            half_vector = span_features.shape[-1]//2
            half_dim = self.lal_d_proj//2
            for i in range(contributions.shape[0]):
                for j in range(contributions.shape[1]):
                    for l in range(contributions.shape[-1]):
                        contributions[i,j,l] = span_features[i,j,l*half_dim:(l+1)*half_dim].sum() + span_features[i,j,half_vector+l*half_dim:half_vector+(l+1)*half_dim].sum()
                    contributions[i,j,:] = (contributions[i,j,:] - np.min(contributions[i,j,:]))
                    contributions[i,j,:] = (contributions[i,j,:])/(np.max(contributions[i,j,:]) - np.min(contributions[i,j,:]))
                    #contributions[i,j,:] = contributions[i,j,:]/np.sum(contributions[i,j,:])
            contributions = torch.softmax(torch.Tensor(contributions), -1)

        label_scores_chart = self.f_label(span_features)
        label_scores_chart = torch.cat([
            label_scores_chart.new_zeros((label_scores_chart.size(0), label_scores_chart.size(1), 1)),
            label_scores_chart
            ], 2)
        if self.contributions and self.use_lal:
            return label_scores_chart, contributions
        return label_scores_chart

    def parse_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end, sentence, sentence_idx, gold=None):
        is_train = gold is not None
        contributions = None
        if self.contributions and self.use_lal:
            label_scores_chart, contributions = self.label_scores_from_annotations(fencepost_annotations_start, fencepost_annotations_end)
        else:
            label_scores_chart = self.label_scores_from_annotations(fencepost_annotations_start, fencepost_annotations_end)
        label_scores_chart_np = label_scores_chart.cpu().data.numpy()

        if is_train:
            decoder_args = dict(
                sentence_len=len(sentence),
                label_scores_chart=label_scores_chart_np,
                gold=gold,
                label_vocab=self.label_vocab,
                is_train=is_train)

            p_score, p_i, p_j, p_label, p_augment = const_decoder.decode(False, **decoder_args)
            g_score, g_i, g_j, g_label, g_augment = const_decoder.decode(True, **decoder_args)
            return p_i, p_j, p_label, p_augment, g_i, g_j, g_label
        else:
            arc_score, type_score = self.dep_score(fencepost_annotations_start, fencepost_annotations_end)

            arc_score_dc = torch.transpose(arc_score, 0, 1)
            arc_dc_np = arc_score_dc.cpu().data.numpy()

            type_np = type_score.cpu().data.numpy()
            type_np = type_np[1:, :]  # remove root
            type = type_np.argmax(axis=1)
            return self.decode_from_chart(sentence, label_scores_chart_np, arc_dc_np, type, sentence_idx=sentence_idx, contributions=contributions)

    def decode_from_chart_batch(self, sentences, charts_np, golds=None):
        trees = []
        scores = []
        if golds is None:
            golds = [None] * len(sentences)
        for sentence, chart_np, gold in zip(sentences, charts_np, golds):
            tree, score = self.decode_from_chart(sentence, chart_np, gold)
            trees.append(tree)
            scores.append(score)
        return trees, scores

    def decode_from_chart(self, sentence, label_scores_chart_np, arc_dc_np, type, sentence_idx=None, gold=None, contributions=None):

        decoder_args = dict(
            sentence_len=len(sentence),
            label_scores_chart= label_scores_chart_np * self.hparams.const_lada,
            type_scores_chart = arc_dc_np * (1.0 - self.hparams.const_lada),
            gold=gold,
            label_vocab=self.label_vocab,
            type_vocab = self.type_vocab,
            is_train=False)

        force_gold = (gold is not None)

        # The optimized cython decoder implementation doesn't actually
        # generate trees, only scores and span indices. When converting to a
        # tree, we assume that the indices follow a preorder traversal.

        score, p_i, p_j, p_label, p_father, p_type, _ = hpsg_decoder.decode(force_gold, **decoder_args)
        if contributions is not None:
            d_l = (self.label_vocab.size - 2)
            mb_size = (self.current_attns.shape[0] // d_l)
            print('SENTENCE', sentence)

        idx = -1
        def make_tree():
            nonlocal idx
            idx += 1
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self.label_vocab.value(label_idx)
            if contributions is not None:
                if label_idx > 0:
                    print(i, sentence[i], j, sentence[j-1], label, label_idx, contributions[i,j,label_idx-1])
                    print("CONTRIBUTIONS")
                    print(list(enumerate(contributions[i,j])))
                    print("ATTENTION DIST")
                    print(torch.softmax(self.current_attns[sentence_idx::mb_size, 0, i:j+1], -1))
            if (i + 1) >= j:
                tag, word = sentence[i]
                if type is not None:
                    tree = trees.LeafParseNode(int(i), tag, word, p_father[i], self.type_vocab.value(type[i]))
                else:
                    tree = trees.LeafParseNode(int(i), tag, word, p_father[i], self.type_vocab.value(p_type[i]))
                if label:
                    assert label[0] != Sub_Head
                    tree = trees.InternalParseNode(label, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if label and label[0] != Sub_Head:
                    return [trees.InternalParseNode(label, children)]
                else:
                    return children

        tree_list = make_tree()
        assert len(tree_list) == 1
        tree = tree_list[0]
        return tree, score
"""
