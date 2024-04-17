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

def unpack_batch(batch, device):
    """ Unpack a batch from the data loader. """
    inputs = [b.to(device) if b is not None else None for b in batch[:4]]
    orig_text = batch[4]
    orig_idx = batch[5]
    return inputs, orig_text, orig_idx

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, emb_matrix=None, model_file=None, device=None):
        if model_file is not None:
            # load from file
            self.load(model_file)
        else:
            self.args = args
            self.model = None if args['dict_only'] else Seq2SeqModel(args, emb_matrix=emb_matrix)
            self.vocab = vocab
            self.expansion_dict = dict()
        if not self.args['dict_only']:
            self.model = self.model.to(device)
            self.crit = loss.SequenceLoss(self.vocab.size).to(device)
            self.optimizer = utils.get_optimizer(self.args['optim'], self.model, self.args['lr'])

    def update(self, batch, eval=False):
        device = next(self.model.parameters()).device
        # ignore the original text when training
        # can try to learn the correct values, even if we eventually
        # copy directly from the original text
        inputs, _, orig_idx = unpack_batch(batch, device)
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
        device = next(self.model.parameters()).device
        inputs, orig_text, orig_idx = unpack_batch(batch, device)
        src, src_mask, tgt, tgt_mask = inputs

        self.model.eval()
        batch_size = src.size(0)
        preds, _ = self.model.predict(src, src_mask, self.args['beam_size'])
        pred_seqs = [self.vocab.unmap(ids) for ids in preds] # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        if self.args.get('force_exact_pieces', False):
            # TODO we should be able to do all this with numpy or something similar
            pred_tokens = []
            for pred_seq, text in zip(pred_seqs, orig_text):
                pred_seq = np.array(list(pred_seq))
                if sum(pred_seq != ' ') == len(text):
                    pred_seq[pred_seq != ' '] = list(text)
                    pred_tokens.append("".join(pred_seq))
                else:
                    pred_tokens.append("".join(pred_seq))
        else:
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

    def dict_expansion(self, word):
        """
        Check the expansion dictionary for the word along with a couple common lowercasings of the word

        (Leadingcase and UPPERCASE)
        """
        expansion = self.expansion_dict.get(word)
        if expansion is not None:
            return expansion

        if word.isupper():
            expansion = self.expansion_dict.get(word.lower())
            if expansion is not None:
                return expansion.upper()

        if word[0].isupper() and word[1:].islower():
            expansion = self.expansion_dict.get(word.lower())
            if expansion is not None:
                return expansion[0].upper() + expansion[1:]

        # could build a truecasing model of some kind to handle cRaZyCaSe...
        # but that's probably too much effort
        return None

    def predict_dict(self, words):
        """ Predict a list of expansions given words. """
        expansions = []
        for w in words:
            expansion = self.dict_expansion(w)
            if expansion is not None:
                expansions.append(expansion)
            else:
                expansions.append(w)
        return expansions

    def ensemble(self, cands, other_preds):
        """ Ensemble the dict with statistical model predictions. """
        expansions = []
        assert len(cands) == len(other_preds)
        for c, pred in zip(cands, other_preds):
            expansion = self.dict_expansion(c)
            if expansion is not None:
                expansions.append(expansion)
            else:
                expansions.append(pred)
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

    def load(self, filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        self.expansion_dict = checkpoint['dict']
        if not self.args['dict_only']:
            self.model = Seq2SeqModel(self.args)
            # could remove strict=False after rebuilding all models,
            # or could switch to 1.6.0 torch with the buffer in seq2seq persistent=False
            self.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            self.model = None
        self.vocab = Vocab.load_state_dict(checkpoint['vocab'])

