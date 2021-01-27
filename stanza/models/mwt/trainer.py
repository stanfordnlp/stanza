"""
A trainer class to handle training and testing of models.
"""

import sys
import numpy as np
from collections import Counter
import logging
import torch
from torch import nn
import torch.nn.init as init

import stanza.models.common.seq2seq_constant as constant
from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.common.seq2seq_model import Seq2SeqModel
from stanza.models.common import utils, loss
from stanza.models.mwt.vocab import Vocab

logger = logging.getLogger('stanza')

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:4]]
    else:
        inputs = [b if b is not None else None for b in batch[:4]]
    orig_idx = batch[4]
    return inputs, orig_idx

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, emb_matrix=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load from file
            self.load(model_file, use_cuda)
        else:
            self.args = args
            self.model = None if args['dict_only'] else Seq2SeqModel(args, emb_matrix=emb_matrix)
            self.vocab = vocab
            self.expansion_dict = dict()
        if not self.args['dict_only']:
            self.crit = loss.SequenceLoss(self.vocab.size)
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
            if use_cuda:
                self.model.cuda()
                self.crit.cuda()
            else:
                self.model.cpu()
                self.crit.cpu()
            self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'])

    def update(self, batch, eval=False):
        inputs, orig_idx = unpack_batch(batch, self.use_cuda)
        src, src_mask, tgt_in, tgt_out = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs, _ = self.model(src, src_mask, tgt_in)
        loss = self.crit(log_probs.view(-1, self.vocab.size), tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx = unpack_batch(batch, self.use_cuda)
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

    def train_dict(self, pairs):
        """ Train a MWT expander given training word-expansion pairs. """
        # accumulate counter
        ctr = Counter()
        ctr.update([(p[0], p[1]) for p in pairs])
        seen = set()
        # find the most frequent mappings
        for p, _ in ctr.most_common():
            w, l = p
            if w not in seen and w != l:
                self.expansion_dict[w] = l
            seen.add(w)
        return

    def predict_dict(self, words):
        """ Predict a list of expansions given words. """
        expansions = []
        for w in words:
            if w in self.expansion_dict:
                expansions += [self.expansion_dict[w]]
            elif w.lower() in self.expansion_dict:
                expansions += [self.expansion_dict[w.lower()]]
            else:
                expansions += [w]
        return expansions

    def ensemble(self, cands, other_preds):
        """ Ensemble the dict with statistical model predictions. """
        expansions = []
        assert len(cands) == len(other_preds)
        for c, pred in zip(cands, other_preds):
            if c in self.expansion_dict:
                expansions += [self.expansion_dict[c]]
            elif c.lower() in self.expansion_dict:
                expansions += [self.expansion_dict[c.lower()]]
            else:
                expansions += [pred]
        return expansions

    def save(self, filename):
        params = {
                'model': self.model.state_dict() if self.model is not None else None,
                'dict': self.expansion_dict,
                'vocab': self.vocab.state_dict(),
                'config': self.args
                }
        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except BaseException:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename, use_cuda=False):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        self.expansion_dict = checkpoint['dict']
        if not self.args['dict_only']:
            self.model = Seq2SeqModel(self.args, use_cuda=use_cuda)
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model = None
        self.vocab = Vocab.load_state_dict(checkpoint['vocab'])

