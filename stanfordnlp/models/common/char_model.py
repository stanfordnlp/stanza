import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence, PackedSequence

from stanfordnlp.models.common.packed_lstm import PackedLSTM
from stanfordnlp.models.common.utils import tensor_unsort

class CharacterModel(nn.Module):
    def __init__(self, args, vocab, pad=False, bidirectional=False, attention=True):
        super().__init__()
        self.args = args
        self.pad = pad
        self.num_dir = 2 if bidirectional else 1
        self.attn = attention

        # char embeddings
        self.char_emb = nn.Embedding(len(vocab['char']), self.args['char_emb_dim'], padding_idx=0)
        if self.attn: 
            self.char_attn = nn.Linear(self.num_dir * self.args['char_hidden_dim'], 1, bias=False)
            self.char_attn.weight.data.zero_()

        # modules
        self.charlstm = PackedLSTM(self.args['char_emb_dim'], self.args['char_hidden_dim'], self.args['char_num_layers'], batch_first=True, \
                dropout=0 if self.args['char_num_layers'] == 1 else args['dropout'], rec_dropout = self.args['char_rec_dropout'], bidirectional=bidirectional)
        self.charlstm_h_init = nn.Parameter(torch.zeros(self.num_dir * self.args['char_num_layers'], 1, self.args['char_hidden_dim']))
        self.charlstm_c_init = nn.Parameter(torch.zeros(self.num_dir * self.args['char_num_layers'], 1, self.args['char_hidden_dim']))

        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, chars, chars_mask, word_orig_idx, sentlens, wordlens):
        embs = self.dropout(self.char_emb(chars))
        batch_size = embs.size(0)
        embs = pack_padded_sequence(embs, wordlens, batch_first=True)
        output = self.charlstm(embs, wordlens, hx=(\
                self.charlstm_h_init.expand(self.num_dir * self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous(), \
                self.charlstm_c_init.expand(self.num_dir * self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous()))
         
        # apply attention, otherwise take final states
        if self.attn:
            char_reps = output[0]
            weights = torch.sigmoid(self.char_attn(self.dropout(char_reps.data)))
            char_reps = PackedSequence(char_reps.data * weights, char_reps.batch_sizes)
            char_reps, _ = pad_packed_sequence(char_reps, batch_first=True)
            res = char_reps.sum(1)
        else:
            h, c = output[1]
            res = h[-2:].transpose(0,1).contiguous().view(batch_size, -1)

        # recover character order and word separation
        res = tensor_unsort(res, word_orig_idx)
        res = pack_sequence(res.split(sentlens))
        if self.pad:
            res = pad_packed_sequence(res, batch_first=True)[0]

        return res
