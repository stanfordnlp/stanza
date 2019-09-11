import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from stanfordnlp.models.common.biaffine import BiaffineScorer
from stanfordnlp.models.common.hlstm import HighwayLSTM
from stanfordnlp.models.common.packed_lstm import PackedLSTM
from stanfordnlp.models.common.dropout import WordDropout
from stanfordnlp.models.common.char_model import CharacterModel, CharacterLanguageModel
from stanfordnlp.models.common.crf import CRFLoss
from stanfordnlp.models.common.vocab import PAD_ID

class NERTagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            self.word_emb = nn.Embedding(len(self.vocab['word']), self.args['word_emb_dim'], PAD_ID)
            # load pretrained embeddings if specified
            if emb_matrix is not None:
                self.init_emb(emb_matrix)
            input_size += self.args['word_emb_dim']

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args['char_context']:
                self.charmodel_forward = CharacterLanguageModel.load(args['charlm_forward_file'])
                self.charmodel_backward = CharacterLanguageModel.load(args['charlm_backward_file'])
                self.charmodel_forward.eval()
                self.charmodel_backward.eval()
            else:
                self.charmodel = CharacterModel(args, vocab, bidirectional=True, attention=False)
            input_size += self.args['char_hidden_dim'] * 2
       
        # recurrent layers
        self.taggerlstm = PackedLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, \
                bidirectional=True, dropout=0 if self.args['num_layers'] == 1 else self.args['dropout'])
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # tag classifier
        num_tag = len(self.vocab['tag'])
        self.tag_clf = nn.Linear(self.args['hidden_dim']*2, num_tag)
        self.tag_clf.bias.data.zero_()

        # criterion
        self.crit = CRFLoss(num_tag, size_average=True)

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def init_emb(self, emb_matrix):
        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = torch.from_numpy(emb_matrix)
        vocab_size = len(self.vocab['word'])
        dim = self.args['word_emb_dim']
        assert emb_matrix.size() == (vocab_size, dim), \
            "Input embedding matrix must match size: {} x {}".format(vocab_size, dim)
        self.word_emb.weight.data.copy_(emb_matrix)

    def forward(self, word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx):
        
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        
        inputs = []
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('char_context', None):
                char_reps_forward = self.charmodel_forward.get_representation(chars[0], charoffsets[0], charlens, char_orig_idx)
                char_reps_forward = PackedSequence(char_reps_forward.data, char_reps_forward.batch_sizes)
                char_reps_backward = self.charmodel_backward.get_representation(chars[1], charoffsets[1], charlens, char_orig_idx)
                char_reps_backward = PackedSequence(char_reps_backward.data, char_reps_backward.batch_sizes)
                inputs += [char_reps_forward, char_reps_backward]
            else:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(char_reps.data, char_reps.batch_sizes)
                inputs += [char_reps]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        if self.args['word_dropout'] > 0:
            lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(\
                self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), \
                self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data


        # prediction layer
        lstm_outputs = self.drop(lstm_outputs)
        logits = pad(self.tag_clf(lstm_outputs)).contiguous()
        loss, trans = self.crit(logits, word_mask, tags)

        return loss, logits, trans
