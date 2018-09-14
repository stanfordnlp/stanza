import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.biaffine import BiaffineScorer
from models.common.hlstm import HighwayLSTM
from models.common.packed_lstm import PackedLSTM
from models.common.utils import tensor_unsort
from models.common.biaffine import BiaffineScorer
from models.common.broadcast_dropout import BroadcastDropout as Dropout

from models.common.vocab import Vocab as BaseVoab
from models.common.vocab import CompositeVocab

class CharacterModel(nn.Module):
    def __init__(self, args, vocab, pad=True):
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

        self.char_dropout = WordDropout(args['char_dropout'])
        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, chars, chars_mask, word_orig_idx, sentlens):
        embs = self.dropout(self.char_dropout(self.char_emb(chars)))
        char_reps = self.dropout(self.charlstm(embs, chars_mask, hx=(self.charlstm_h_init.expand(self.args['char_num_layers'], embs.size(0), self.args['char_hidden_dim']).contiguous(), self.charlstm_c_init.expand(self.args['char_num_layers'], embs.size(0), self.args['char_hidden_dim']).contiguous()))[0])

        # attention
        weights = F.sigmoid(self.char_attn(char_reps)).masked_fill(chars_mask.unsqueeze(2), 0)
        weights = weights.transpose(1, 2)

        res = weights.bmm(char_reps).squeeze(1)
        res = tensor_unsort(res, word_orig_idx)

        res = nn.utils.rnn.pack_sequence(res.split(sentlens))
        if self.pad:
            res = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)[0]

        return res

class WordDropout(nn.Module):
    def __init__(self, dropprob):
        super().__init__()
        self.dropprob = dropprob

    def forward(self, x, replacement=None):
        if not self.training or self.dropprob == 0:
            return x

        masksize = [y for y in x.size()]
        masksize[-1] = 1
        dropmask = torch.rand(*masksize, device=x.device) < self.dropprob

        res = x.masked_fill(dropmask, 0)
        if replacement is not None:
            res = res + dropmask.float() * replacement

        return res

class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid

        # pretrained embeddings
        self.pretrained_emb = nn.Embedding(emb_matrix.shape[0], emb_matrix.shape[1], padding_idx=0)
        self.pretrained_emb.from_pretrained(torch.from_numpy(emb_matrix), freeze=True)

        # frequent word embeddings
        self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)

        if not share_hid:
            # upos embeddings
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

        # modules
        if self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'])
            input_size = self.args['word_emb_dim'] + self.args['transformed_dim'] * 2 # freq word + transformed pretrained + transformed char-level
        else:
            input_size = self.args['word_emb_dim'] + self.args['transformed_dim']
        self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)

        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=F.tanh)
        self.drop_replacement = nn.Parameter(torch.zeros(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        if share_hid:
            clf_constructor = lambda outsize: nn.Linear(self.args['deep_biaff_hidden_dim'], outsize)
        else:
            self.xpos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
            self.ufeats_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
            clf_constructor = lambda outsize: BiaffineScorer(self.args['deep_biaff_hidden_dim'], self.args['tag_emb_dim'], outsize)

        if isinstance(vocab['xpos'], CompositeVocab):
            self.xpos_clf = nn.ModuleList()
            for l in vocab['xpos'].lens():
                self.xpos_clf.append(clf_constructor(l))
        else:
            self.xpos_clf = clf_constructor(len(vocab['xpos']))
            self.xpos_clf.weight.data.zero_()
            self.xpos_clf.bias.data.zero_()

        self.ufeats_clf = nn.ModuleList()
        for l in vocab['feats'].lens():
            self.ufeats_clf.append(clf_constructor(l))
            if share_hid:
                self.ufeats_clf[-1].weight.data.zero_()
                self.ufeats_clf[-1].bias.data.zero_()

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=0) # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens):
        pretrained_emb = self.pretrained_emb(pretrained)
        pretrained_emb = self.trans_pretrained(pretrained_emb)

        word_emb = self.word_emb(word)

        if self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens)
            char_reps = self.trans_char(self.drop(char_reps))

            lstm_inputs = torch.cat([char_reps, pretrained_emb, word_emb], 2)
        else:
            lstm_inputs = torch.cat([pretrained_emb, word_emb], 2)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)

        lstm_outputs = (self.taggerlstm(lstm_inputs, word_mask, hx=(self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous())))

        upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        upos_pred = self.upos_clf(self.drop(upos_hid))

        preds = [upos_pred.max(2)[1]]

        loss = self.crit(upos_pred.view(-1, upos_pred.size(2)), upos.view(-1))

        if self.share_hid:
            xpos_hid = upos_hid
            ufeats_hid = upos_hid

            clffunc = lambda clf, hid: clf(self.drop(hid))
        else:
            xpos_hid = F.relu(self.xpos_hid(self.drop(lstm_outputs)))
            ufeats_hid = F.relu(self.ufeats_hid(self.drop(lstm_outputs)))

            if self.training:
                upos_emb = self.upos_emb(upos)
            else:
                upos_emb = self.upos_emb(preds[-1])

            clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(upos_emb))

        if isinstance(self.vocab['xpos'], CompositeVocab):
            xpos_preds = []
            for i in range(len(self.vocab['xpos'])):
                xpos_pred = clffunc(self.xpos_clf[i], xpos_hid)
                loss += self.crit(xpos_pred.view(-1, xpos_pred.size(2)), xpos[:, :, i].view(-1))
                xpos_preds.append(xpos_pred.max(2, keepdim=True)[1])
            preds.append(torch.cat(xpos_preds, 2))
        else:
            xpos_pred = clffunc(self.xpos_clf, xpos_hid)
            loss += self.crit(xpos_pred.view(-1, xpos_pred.size(2)), xpos.view(-1))
            preds.append(xpos_pred.max(2)[1])

        ufeats_preds = []
        for i in range(len(self.vocab['feats'])):
            ufeats_pred = clffunc(self.ufeats_clf[i], ufeats_hid)
            loss += self.crit(ufeats_pred.view(-1, ufeats_pred.size(2)), ufeats[:, :, i].view(-1))
            ufeats_preds.append(ufeats_pred.max(2, keepdim=True)[1])
        preds.append(torch.cat(ufeats_preds, 2))

        return loss, preds
