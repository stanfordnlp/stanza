"""
A trainer class to handle training and testing of models.
"""

import sys
import logging
import torch
from torch import nn

from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.common import utils, loss
from stanza.models.ner.model import NERTagger
from stanza.models.ner.vocab import MultiVocab
from stanza.models.common.crf import viterbi_decode

logger = logging.getLogger('stanza')

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:6]]
    else:
        inputs = batch[:6]
    orig_idx = batch[6]
    word_orig_idx = batch[7]
    char_orig_idx = batch[8]
    sentlens = batch[9]
    wordlens = batch[10]
    charlens = batch[11]
    charoffsets = batch[12]
    return inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets

def fix_singleton_tags(tags):
    """
    If there are any singleton B- tags, convert them to S-
    """
    new_tags = list(tags)
    for idx, tag in enumerate(new_tags):
        if (tag.startswith("B-") and
            (idx == len(new_tags) - 1 or
             (new_tags[idx+1] != "I-" + tag[2:] and new_tags[idx+1] != "E-" + tag[2:]))):
            new_tags[idx] = "S-" + tag[2:]
    return new_tags

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False,
                 train_classifier_only=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file, args)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = NERTagger(args, vocab, emb_matrix=pretrain.emb)

        if train_classifier_only:
            logger.info('Disabling gradient for non-classifier layers')
            exclude = ['tag_clf', 'crit']
            for pname, p in self.model.named_parameters():
                if pname.split('.')[0] not in exclude:
                    p.requires_grad = False
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], momentum=self.args['momentum'])

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, chars, tags = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _, _ = self.model(word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, chars, tags = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, logits, trans = self.model(word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx)

        # decode
        trans = trans.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :sentlens[i]], trans)
            tags = self.vocab['tag'].unmap(tags)
            tags = fix_singleton_tags(tags)
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
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename, args=None):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        if args: self.args.update(args)
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.model = NERTagger(self.args, self.vocab)
        self.model.load_state_dict(checkpoint['model'], strict=False)

