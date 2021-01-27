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
from stanza.models.common.seq2seq_model import Seq2SeqModel
from stanza.models.common import utils, loss
from stanza.models.lemma import edit
from stanza.models.lemma.vocab import MultiVocab

logger = logging.getLogger('stanza')

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:6]]
    else:
        inputs = [b if b is not None else None for b in batch[:6]]
    orig_idx = batch[6]
    return inputs, orig_idx

class Trainer(object):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, emb_matrix=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file, use_cuda)
        else:
            # build model from scratch
            self.args = args
            self.model = None if args['dict_only'] else Seq2SeqModel(args, emb_matrix=emb_matrix, use_cuda=use_cuda)
            self.vocab = vocab
            # dict-based components
            self.word_dict = dict()
            self.composite_dict = dict()
        if not self.args['dict_only']:
            if self.args.get('edit', False):
                self.crit = loss.MixLoss(self.vocab['char'].size, self.args['alpha'])
                logger.debug("Running seq2seq lemmatizer with edit classifier...")
            else:
                self.crit = loss.SequenceLoss(self.vocab['char'].size)
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
        src, src_mask, tgt_in, tgt_out, pos, edits = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs, edit_logits = self.model(src, src_mask, tgt_in, pos)
        if self.args.get('edit', False):
            assert edit_logits is not None
            loss = self.crit(log_probs.view(-1, self.vocab['char'].size), tgt_out.view(-1), \
                    edit_logits, edits)
        else:
            loss = self.crit(log_probs.view(-1, self.vocab['char'].size), tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, beam_size=1):
        inputs, orig_idx = unpack_batch(batch, self.use_cuda)
        src, src_mask, tgt, tgt_mask, pos, edits = inputs

        self.model.eval()
        batch_size = src.size(0)
        preds, edit_logits = self.model.predict(src, src_mask, pos=pos, beam_size=beam_size)
        pred_seqs = [self.vocab['char'].unmap(ids) for ids in preds] # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs] # join chars to be tokens
        pred_tokens = utils.unsort(pred_tokens, orig_idx)
        if self.args.get('edit', False):
            assert edit_logits is not None
            edits = np.argmax(edit_logits.data.cpu().numpy(), axis=1).reshape([batch_size]).tolist()
            edits = utils.unsort(edits, orig_idx)
        else:
            edits = None
        return pred_tokens, edits

    def postprocess(self, words, preds, edits=None):
        """ Postprocess, mainly for handing edits. """
        assert len(words) == len(preds), "Lemma predictions must have same length as words."
        edited = []
        if self.args.get('edit', False):
            assert edits is not None and len(words) == len(edits)
            for w, p, e in zip(words, preds, edits):
                lem = edit.edit_word(w, p, e)
                edited += [lem]
        else:
            edited = preds # do not edit
        # final sanity check
        assert len(edited) == len(words)
        final = []
        for lem, w in zip(edited, words):
            if len(lem) == 0 or constant.UNK in lem:
                final += [w] # invalid prediction, fall back to word
            else:
                final += [lem]
        return final

    def update_lr(self, new_lr):
        utils.change_lr(self.optimizer, new_lr)

    def train_dict(self, triples):
        """ Train a dict lemmatizer given training (word, pos, lemma) triples. """
        # accumulate counter
        ctr = Counter()
        ctr.update([(p[0], p[1], p[2]) for p in triples])
        # find the most frequent mappings
        for p, _ in ctr.most_common():
            w, pos, l = p
            if (w,pos) not in self.composite_dict:
                self.composite_dict[(w,pos)] = l
            if w not in self.word_dict:
                self.word_dict[w] = l
        return

    def predict_dict(self, pairs):
        """ Predict a list of lemmas using the dict model given (word, pos) pairs. """
        lemmas = []
        for p in pairs:
            w, pos = p
            if (w,pos) in self.composite_dict:
                lemmas += [self.composite_dict[(w,pos)]]
            elif w in self.word_dict:
                lemmas += [self.word_dict[w]]
            else:
                lemmas += [w]
        return lemmas

    def skip_seq2seq(self, pairs):
        """ Determine if we can skip the seq2seq module when ensembling with the frequency lexicon. """

        skip = []
        for p in pairs:
            w, pos = p
            if (w,pos) in self.composite_dict:
                skip.append(True)
            elif w in self.word_dict:
                skip.append(True)
            else:
                skip.append(False)
        return skip

    def ensemble(self, pairs, other_preds):
        """ Ensemble the dict with statistical model predictions. """
        lemmas = []
        assert len(pairs) == len(other_preds)
        for p, pred in zip(pairs, other_preds):
            w, pos = p
            if (w,pos) in self.composite_dict:
                lemma = self.composite_dict[(w,pos)]
            elif w in self.word_dict:
                lemma = self.word_dict[w]
            else:
                lemma = pred
            if lemma is None:
                lemma = w
            lemmas.append(lemma)
        return lemmas

    def save(self, filename):
        params = {
                'model': self.model.state_dict() if self.model is not None else None,
                'dicts': (self.word_dict, self.composite_dict),
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
        self.word_dict, self.composite_dict = checkpoint['dicts']
        if not self.args['dict_only']:
            self.model = Seq2SeqModel(self.args, use_cuda=use_cuda)
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model = None
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
