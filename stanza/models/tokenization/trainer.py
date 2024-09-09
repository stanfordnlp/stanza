import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from stanza.models.common import utils
from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.tokenization.utils import create_dictionary

from .model import Tokenizer
from .vocab import Vocab

logger = logging.getLogger('stanza')

class Trainer(BaseTrainer):
    def __init__(self, args=None, vocab=None, lexicon=None, dictionary=None, model_file=None, device=None, pretrain=None):
        self.pretrain = pretrain
        if model_file is not None:
            # load everything from file
            self.load(model_file)
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.lexicon = lexicon
            self.dictionary = dictionary
            self.model = Tokenizer(self.args, self.args['vocab_size'], self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'], feat_dropout=self.args['feat_dropout'], pretrain=pretrain)

        if self.args["sentence_second_pass"]:
            assert bool(pretrain), "context-aware sentence analysis requires pretrained wordvectors; download them!"

        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        self.optimizer = utils.get_optimizer("adam", self.model, lr=self.args['lr0'], betas=(.9, .9), weight_decay=self.args['weight_decay'])
        self.feat_funcs = self.args.get('feat_funcs', None)
        self.lang = self.args['lang'] # language determines how token normalization is done
        self.pretrain = pretrain
        self.global_step_counter_ = 0
        self.train_2nd_pass = False

    @property
    def steps(self):
        return self.global_step_counter_

    def update(self, inputs):
        self.global_step_counter_ += 1
        self.model.train()
        units, labels, features, text = inputs

        device = next(self.model.parameters()).device
        units = units.to(device)
        labels = labels.to(device)
        features = features.to(device)

        # we detach 2nd pass if we are not training second pass
        pred = self.model(units, features, text, not self.train_2nd_pass)

        self.optimizer.zero_grad()
        classes = pred.size(2)
        loss = self.criterion(pred.view(-1, classes), labels.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()

        return loss.item()

    def predict(self, inputs):
        self.model.eval()
        units, _, features, text = inputs

        device = next(self.model.parameters()).device
        units = units.to(device)
        features = features.to(device)

        pred = self.model(units, features, text)

        return pred.data.cpu().numpy()

    def save(self, filename):
        params = {
            'model': self.model.state_dict() if self.model is not None else None,
            'vocab': self.vocab.state_dict(),
            'lexicon': self.lexicon,
            'config': self.args,
            'steps': self.global_step_counter_
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
        if self.args.get('use_mwt', None) is None:
            # Default to True as many currently saved models
            # were built with mwt layers
            self.args['use_mwt'] = True
        self.model = Tokenizer(self.args, self.args['vocab_size'], self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'], feat_dropout=self.args['feat_dropout'], pretrain=self.pretrain)
        self.model.load_state_dict(checkpoint['model'])
        self.vocab = Vocab.load_state_dict(checkpoint['vocab'])
        self.lexicon = checkpoint['lexicon']

        self.global_step_counter_ = checkpoint.get("steps", 0)

        if self.lexicon is not None:
            self.dictionary = create_dictionary(self.lexicon)
        else:
            self.dictionary = None
