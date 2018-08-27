import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.biaffine import BiaffineScorer
from models.common.hlstm import HighwayLSTM
from models.common.packed_lstm import PackedLSTM

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

        # char embeddings
        self.char_emb = nn.Embedding(len(vocab['char']), self.args['char_emb_dim'], padding_idx=0)

        # modules
        self.charlstm = PackedLSTM(self.args['char_emb_dim'], self.args['char_hidden_dim'], self.args['char_num_layers'], batch_first=True)
        self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'])
        self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'])

        input_size = self.args['word_emb_dim'] + self.args['transformed_dim'] * 2 # freq word + transformed pretrained + transformed char-level
        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectinal=True)

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained):

        pass
