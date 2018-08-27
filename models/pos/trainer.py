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
        inputs = [Variable(b.cuda()) if b is not None else None for b in batch[:8]]
    else:
        inputs = [Variable(b) if b is not None else None for b in batch[:8]]
    orig_idx = batch[8]
    word_orig_idx = batch[8]
    return inputs, orig_idx, word_orig_idx

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args, vocab, emb_matrix):
        self.args = args

        self.model = Tagger(args, vocab, emb_matrix=emb_matrix)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if args['cuda']:
            self.model.cuda()
            self.crit.cuda()
        self.optimizer = utils.get_optimizer(args['optim'], self.parameters, args['lr'])

        self.vocab = vocab

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx = unpack_batch(batch, self.args)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx = unpack_batch(batch, self.args)
        src, src_mask, tgt, tgt_mask = inputs

        self.model.eval()
        batch_size = src.size(0)
        preds, _ = self.model.predict(src, src_mask, self.args['beam_size'])
        pred_seqs = [self.vocab.unmap(ids) for ids in preds] # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs] # join chars to be tokens
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

class DictTrainer(BaseTrainer):
    """ A trainer wrapper for a simple dictionary-based MWT expander. """
    def __init__(self, args, vocab=None):
        self.model = dict()

    def train(self, pairs):
        """ Train a MWT expander given training word-expansion pairs. """
        # accumulate counter
        ctr = Counter()
        ctr.update([(p[0], p[1]) for p in pairs])
        seen = set()
        # find the most frequent mappings
        for p, _ in ctr.most_common():
            w, l = p
            if w not in seen and w != l:
                self.model[w] = l
            seen.add(w)
        return

    def predict(self, words):
        """ Predict a list of expansions given words. """
        expansions = []
        for w in words:
            if w in self.model:
                expansions += [self.model[w]]
            elif w.lower() in self.model:
                expansions += [self.model[w.lower()]]
            else:
                expansions += [w]
        return expansions

    def save(self, filename):
        params = {
                'model': self.model,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model = checkpoint['model']

    def ensemble(self, cands, other_preds):
        """ Ensemble the dict with another model predictions. """
        expansions = []
        assert len(cands) == len(other_preds)
        for c, pred in zip(cands, other_preds):
            if c in self.model:
                expansions += [self.model[c]]
            elif c.lower() in self.model:
                expansions += [self.model[c.lower()]]
            else:
                expansions += [pred]
        return expansions
