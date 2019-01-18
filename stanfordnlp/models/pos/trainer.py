"""
A trainer class to handle training and testing of models.
"""

import torch
from torch import nn

from stanfordnlp.models.common.trainer import Trainer as BaseTrainer
from stanfordnlp.models.common import utils, loss
from stanfordnlp.models.pos.model import Tagger
from stanfordnlp.models.pos.vocab import MultiVocab

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
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
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(pretrain, model_file)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = Tagger(args, vocab, emb_matrix=pretrain.emb, share_hid=args['share_hid'])
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6)

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
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
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
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

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'vocab': self.vocab.state_dict(),
                'config': self.args
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, pretrain, filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.args = checkpoint['config']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.model = Tagger(self.args, self.vocab, emb_matrix=pretrain.emb, share_hid=self.args['share_hid'])
        self.model.load_state_dict(checkpoint['model'])

