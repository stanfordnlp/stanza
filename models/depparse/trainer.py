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
from models.common.chuliu_edmonds import chuliu_edmonds_one_root

from models.depparse.model import Parser

def unpack_batch(batch, args):
    """ Unpack a batch from the data loader. """
    if args['cuda']:
        inputs = [b.cuda() if b is not None else None for b in batch[:11]]
    else:
        inputs = batch[:11]
    orig_idx = batch[11]
    word_orig_idx = batch[12]
    sentlens = batch[13]
    wordlens = batch[14]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args, vocab, emb_matrix=None):
        self.args = args

        self.model = Parser(args, vocab, emb_matrix=emb_matrix)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if args['cuda']:
            self.model.cuda()
        self.optimizer = utils.get_optimizer(args['optim'], self.parameters, args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6)

        self.vocab = vocab

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.args)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.args)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)
        head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], sentlens)] # remove attachment for the root
        deprel_seqs = [self.vocab['deprel'].unmap([preds[1][i][j+1][h] for j, h in enumerate(hs)]) for i, hs in enumerate(head_seqs)]

        pred_tokens = [[[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(sentlens[i]-1)] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens
