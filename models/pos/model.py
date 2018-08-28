import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.biaffine import BiaffineScorer
from models.common.hlstm import HighwayLSTM
from models.common.packed_lstm import PackedLSTM
from models.common.utils import unsort

class CharacterModel(nn.Module):
    def __init__(self, args, vocab, pad=True):
        super().__init__()
        self.args = args
        self.pad = pad
        # char embeddings
        self.char_emb = nn.Embedding(len(vocab['char']), self.args['char_emb_dim'], padding_idx=0)
        self.char_attn = nn.Linear(self.args['char_hidden_dim'], 1)

        # modules
        self.charlstm = PackedLSTM(self.args['char_emb_dim'], self.args['char_hidden_dim'], self.args['char_num_layers'], batch_first=True, pad=True)

    def forward(self, chars, chars_mask, word_orig_idx, sentlens):
        embs = self.char_emb(chars)
        char_reps, _ = self.charlstm(embs, chars_mask)

        # attention
        weights = F.sigmoid(self.char_attn(char_reps).masked_fill(chars_mask.unsqueeze(2), -float('inf')))
        weights = weights / weights.sum(1, keepdim=True) + 1e-20
        weights = weights.transpose(1, 2)

        res = weights.bmm(char_reps).squeeze(1)

        res = nn.utils.rnn.pack_sequence(res.split(sentlens))
        if self.pad:
            res = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)[0]

        return res

class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix):
        super().__init__()

        self.vocab = vocab
        self.args = args

        # pretrained embeddings
        self.pretrained_emb = nn.Embedding(emb_matrix.shape[0], emb_matrix.shape[1], padding_idx=0)
        self.pretrained_emb.from_pretrained(torch.from_numpy(emb_matrix), freeze=True)

        # frequent word embeddings
        self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)

        # upos embeddings
        self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

        # modules
        self.charmodel = CharacterModel(args, vocab)
        self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'])
        self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'])

        input_size = self.args['word_emb_dim'] + self.args['transformed_dim'] * 2 # freq word + transformed pretrained + transformed char-level
        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True)

        # classifiers
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))

        # criteria
        self.upos_crit = nn.CrossEntropyLoss(ignore_index=0) # ignore padding

        self.drop = nn.Dropout(args['dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens):
        char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens)
        pretrained_emb = self.pretrained_emb(pretrained)
        word_emb = self.word_emb(word)

        pretrained_emb = self.trans_pretrained(pretrained_emb)
        char_reps = self.trans_char(char_reps)

        lstm_inputs = torch.cat([char_reps, pretrained_emb, word_emb], 2)

        lstm_outputs = self.taggerlstm(lstm_inputs, word_mask)

        upos_hid = self.drop(F.relu(self.upos_hid(lstm_outputs)))
        upos_pred = self.upos_clf(upos_hid)

        loss = self.upos_crit(upos_pred.view(-1, upos_pred.size(2)), upos.view(-1))

        return loss, (upos_pred,)
