import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence, PackedSequence

from stanza.models.common.packed_lstm import PackedLSTM
from stanza.models.common.utils import tensor_unsort, unsort
from stanza.models.common.dropout import SequenceUnitDropout
from stanza.models.common.vocab import UNK_ID, CharVocab

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

class CharacterLanguageModel(nn.Module):

    def __init__(self, args, vocab, pad=False, is_forward_lm=True):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.is_forward_lm = is_forward_lm
        self.pad = pad
        self.finetune = True # always finetune unless otherwise specified

        # char embeddings
        self.char_emb = nn.Embedding(len(self.vocab['char']), self.args['char_emb_dim'], padding_idx=None) # we use space as padding, so padding_idx is not necessary
        
        # modules
        self.charlstm = PackedLSTM(self.args['char_emb_dim'], self.args['char_hidden_dim'], self.args['char_num_layers'], batch_first=True, \
                dropout=0 if self.args['char_num_layers'] == 1 else args['char_dropout'], rec_dropout = self.args['char_rec_dropout'], bidirectional=False)
        self.charlstm_h_init = nn.Parameter(torch.zeros(self.args['char_num_layers'], 1, self.args['char_hidden_dim']))
        self.charlstm_c_init = nn.Parameter(torch.zeros(self.args['char_num_layers'], 1, self.args['char_hidden_dim']))

        # decoder
        self.decoder = nn.Linear(self.args['char_hidden_dim'], len(self.vocab['char']))
        self.dropout = nn.Dropout(args['char_dropout'])
        self.char_dropout = SequenceUnitDropout(args.get('char_unit_dropout', 0), UNK_ID)

    def forward(self, chars, charlens, hidden=None):
        chars = self.char_dropout(chars)
        embs = self.dropout(self.char_emb(chars))
        batch_size = embs.size(0)
        embs = pack_padded_sequence(embs, charlens, batch_first=True)
        if hidden is None: 
            hidden = (self.charlstm_h_init.expand(self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous(),
                      self.charlstm_c_init.expand(self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous())
        output, hidden = self.charlstm(embs, charlens, hx=hidden)
        output = self.dropout(pad_packed_sequence(output, batch_first=True)[0])
        decoded = self.decoder(output)
        return output, hidden, decoded

    def get_representation(self, chars, charoffsets, charlens, char_orig_idx):
        with torch.no_grad():
            output, _, _ = self.forward(chars, charlens)
            res = [output[i, offsets] for i, offsets in enumerate(charoffsets)]
            res = unsort(res, char_orig_idx)
            res = pack_sequence(res)
            if self.pad:
                res = pad_packed_sequence(res, batch_first=True)[0]
        return res

    def hidden_dim(self):
        return self.args['char_hidden_dim']

    def char_vocab(self):
        return self.vocab['char']

    def train(self, mode=True):
        """
        Override the default train() function, so that when self.finetune == False, the training mode 
        won't be impacted by the parent models' status change.
        """
        if not mode: # eval() is always allowed, regardless of finetune status
            super().train(mode)
        else:
            if self.finetune: # only set to training mode in finetune status
                super().train(mode)

    def save(self, filename):
        state = {
            'vocab': self.vocab['char'].state_dict(),
            'args': self.args,
            'state_dict': self.state_dict(),
            'pad': self.pad,
            'is_forward_lm': self.is_forward_lm
        }
        torch.save(state, filename, _use_new_zipfile_serialization=False)

    @classmethod
    def load(cls, filename, finetune=False):
        state = torch.load(filename, lambda storage, loc: storage)
        vocab = {'char': CharVocab.load_state_dict(state['vocab'])}
        model = cls(state['args'], vocab, state['pad'], state['is_forward_lm'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.finetune = finetune # set finetune status
        return model
