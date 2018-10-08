import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from models.common.packed_lstm import PackedLSTM
from models.common.utils import tensor_unsort

class CharacterModel(nn.Module):
    def __init__(self, args, vocab, pad=False):
        super().__init__()
        self.args = args
        self.pad = pad
        # char embeddings
        self.char_emb = nn.Embedding(len(vocab['char']), self.args['char_emb_dim'], padding_idx=0)
        self.char_attn = nn.Linear(self.args['char_hidden_dim'], 1, bias=False)
        self.char_attn.weight.data.zero_()

        # modules
        self.charlstm = PackedLSTM(self.args['char_emb_dim'], self.args['char_hidden_dim'], self.args['char_num_layers'], batch_first=True, pad=True, dropout=0 if self.args['char_num_layers'] == 1 else args['dropout'], rec_dropout = self.args['char_rec_dropout'])
        self.charlstm_h_init = nn.Parameter(torch.zeros(self.args['char_num_layers'], 1, self.args['char_hidden_dim']))
        self.charlstm_c_init = nn.Parameter(torch.zeros(self.args['char_num_layers'], 1, self.args['char_hidden_dim']))

        self.dropout = nn.Dropout(args['dropout'], inplace=True)

    def forward(self, chars, chars_mask, word_orig_idx, sentlens, wordlens):
        embs = self.dropout(self.char_emb(chars))
        char_reps = self.charlstm(embs, wordlens, hx=(self.charlstm_h_init.expand(self.args['char_num_layers'], embs.size(0), self.args['char_hidden_dim']).contiguous(), self.charlstm_c_init.expand(self.args['char_num_layers'], embs.size(0), self.args['char_hidden_dim']).contiguous()))[0]

        # attention
        weights = torch.sigmoid(self.char_attn(self.dropout(char_reps))).masked_fill(chars_mask.unsqueeze(2), 0)
        weights = weights.transpose(1, 2)

        res = weights.bmm(char_reps).squeeze(1)
        res = tensor_unsort(res, word_orig_idx)

        res = pack_sequence(res.split(sentlens))
        if self.pad:
            res = pad_packed_sequence(res, batch_first=True)[0]

        return res
