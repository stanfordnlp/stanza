"""
A trainer class to handle training and testing of models.
"""

import numpy as np
from collections import Counter
import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

from models.common.constant import lcode2lang
from models.common.trainer import Trainer as BaseTrainer
from models.common.seq2seq_model import Seq2SeqModel
from models.common import utils, loss

from models.pos.model import Tagger

def unpack_batch(batch, args):
    """ Unpack a batch from the data loader. """
    if args['cuda']:
        inputs = [b.cuda() if b is not None else None for b in batch[:8]]
    else:
        inputs = batch[:8]
    orig_idx = batch[8]
    word_orig_idx = batch[9]
    sentlens = batch[10]
    wordlens = batch[11]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args, vocab, emb_matrix=None):
        self.args = args

        self.model = Tagger(args, vocab, emb_matrix=emb_matrix)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if args['cuda']:
            self.model.cuda()
        self.optimizer = utils.get_optimizer(args['optim'], self.parameters, args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6)

        self.vocab = vocab

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.args)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.args)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens)
        upos_seqs = [self.vocab['upos'].unmap(sent) for sent in preds[0]]
        xpos_seqs = [self.vocab['xpos'].unmap(sent) for sent in preds[1]]
        feats_seqs = [self.vocab['feats'].unmap(sent) for sent in preds[2]]

        pred_tokens = [[[upos_seqs[i][j], xpos_seqs[i][j], feats_seqs[i][j]] for j in range(sentlens[i])] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens
