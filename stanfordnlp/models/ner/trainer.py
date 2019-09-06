"""
A trainer class to handle training and testing of models.
"""

import sys
import logging
import torch
from torch import nn

from stanfordnlp.models.common.trainer import Trainer as BaseTrainer
from stanfordnlp.models.common import utils, loss
from stanfordnlp.models.ner.model import NERTagger
from stanfordnlp.models.ner.vocab import MultiVocab
from stanfordnlp.models.common.crf import viterbi_decode

logger = logging.getLogger(__name__)

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:5]]
    else:
        inputs = batch[:5]
    orig_idx = batch[5]
    word_orig_idx = batch[6]
    sentlens = batch[7]
    wordlens = batch[8]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = NERTagger(args, vocab, emb_matrix=pretrain.emb)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], momentum=self.args['momentum'])

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, tags = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _, _ = self.model(word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, tags = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, logits, trans = self.model(word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens)

        # decode
        trans = trans.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :sentlens[i]], trans)
            tags = self.vocab['tag'].unmap(tags)
            tag_seqs += [tags]

        if unsort:
            tag_seqs = utils.unsort(tag_seqs, orig_idx)
        return tag_seqs

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
                'model': model_state,
                'vocab': self.vocab.state_dict(),
                'config': self.args
                }
        try:
            torch.save(params, filename)
            logger.info("Model saved to {}".format(filename))
        except BaseException:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from {}".format(filename))
            sys.exit(1)
        self.args = checkpoint['config']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.model = NERTagger(self.args, self.vocab)
        self.model.load_state_dict(checkpoint['model'], strict=False)

