import sys
import torch
import torch.nn as nn
import torch.optim as optim

from stanfordnlp.models.common.trainer import Trainer

from .model import Tokenizer
from .vocab import Vocab

class Trainer(Trainer):
    def __init__(self, args=None, vocab=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file)
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = Tokenizer(self.args, self.args['vocab_size'], self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        if use_cuda:
            self.model.cuda()
            self.criterion.cuda()
        else:
            self.model.cpu()
            self.criterion.cpu()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(self.parameters, lr=self.args['lr0'], betas=(.9, .9), weight_decay=self.args['weight_decay'])
        self.feat_funcs = self.args.get('feat_funcs', None)
        self.lang = self.args['lang'] # language determines how token normlization is done

    def update(self, inputs):
        self.model.train()
        units, labels, features, _ = inputs

        if self.use_cuda:
            units = units.cuda()
            labels = labels.cuda()
            features = features.cuda()

        pred = self.model(units, features)

        self.optimizer.zero_grad()
        classes = pred.size(2)
        loss = self.criterion(pred.view(-1, classes), labels.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()

        return loss.item()

    def predict(self, inputs):
        self.model.eval()
        units, labels, features, _ = inputs

        if self.use_cuda:
            units = units.cuda()
            labels = labels.cuda()
            features = features.cuda()

        pred = self.model(units, features)

        return pred.data.cpu().numpy()

    def save(self, filename):
        params = {
                'model': self.model.state_dict() if self.model is not None else None,
                'vocab': self.vocab.state_dict(),
                'config': self.args
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
            sys.exit(1)
        self.args = checkpoint['config']
        self.model = Tokenizer(self.args, self.args['vocab_size'], self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'])
        self.model.load_state_dict(checkpoint['model'])
        self.vocab = Vocab.load_state_dict(checkpoint['vocab'])
