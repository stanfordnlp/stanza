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

import models.common.seq2seq_constant as constant
from models.common.seq2seq_model import Seq2SeqModel
from models.common import utils, loss

def unpack_batch(batch, args):
    """ Unpack a batch from the data loader. """
    if args['cuda']:
        inputs = [Variable(b.cuda()) if b is not None else None for b in batch[:5]]
    else:
        inputs = [Variable(b) if b is not None else None for b in batch[:5]]
    orig_idx = batch[5]
    return inputs, orig_idx

class Trainer(object):
    """ A trainer for training models. """
    def __init__(self, args, vocab, emb_matrix=None):
        self.args = args
        self.model = Seq2SeqModel(args, emb_matrix=emb_matrix)
        self.crit = loss.SequenceLoss(vocab.size)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if args['cuda']:
            self.model.cuda()
            self.crit.cuda()
        self.optimizer = utils.get_optimizer(args['optim'], self.parameters, args['lr'])
        self.vocab = vocab

    def update(self, batch, eval=False):
        inputs, orig_idx = unpack_batch(batch, self.args)
        src, src_mask, tgt_in, tgt_out, pos = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs = self.model(src, src_mask, tgt_in, pos)
        loss = self.crit(log_probs.view(-1, self.vocab.size), tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, beam_size=1, unsort=True):
        inputs, orig_idx = unpack_batch(batch, self.args)
        src, src_mask, tgt, tgt_mask, pos = inputs

        self.model.eval()
        batch_size = src.size(0)
        preds = self.model.predict(src, src_mask, pos=pos, beam_size=beam_size)
        pred_seqs = [self.vocab.unmap(ids) for ids in preds] # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs] # join chars to be tokens
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

    def update_lr(self, new_lr):
        utils.change_lr(self.optimizer, new_lr)

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
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
        self.model.load_state_dict(checkpoint['model'])
        if self.args['mode'] == 'train':
            self.optimizer.load_state_dict(checkpoint['optim'])

class DictTrainer(object):
    """ A trainer wrapper for a simple dictionary-based lemmatizer. """
    def __init__(self, args):
        self.model = dict()

    def train(self, triples):
        """ Train a lemmatizer given training (word, pos, lemma) triples. """
        # accumulate counter
        ctr = Counter()
        ctr.update([(p[0], p[1], p[2]) for p in triples])
        # find the most frequent mappings
        for p, _ in ctr.most_common():
            w, pos, l = p
            if (w,pos) not in self.model:
                self.model[(w,pos)] = l
        return

    def predict(self, pairs):
        """ Predict a list of lemmas given (word, pos) pairs. """
        lemmas = []
        for p in pairs:
            w, pos = p
            if (w,pos) in self.model:
                lemmas += [self.model[(w,pos)]]
            else:
                lemmas += [w]
        return lemmas

    def ensemble(self, pairs, other_preds):
        """ Ensemble the dict with another model predictions. """
        lemmas = []
        assert len(pairs) == len(other_preds)
        for p, pred in zip(pairs, other_preds):
            w, pos = p
            if (w,pos) in self.model:
                lemmas += [self.model[(w,pos)]]
            else:
                lemmas += [pred]
        return lemmas

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

