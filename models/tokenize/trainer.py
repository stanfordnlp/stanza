import torch
import torch.nn as nn
import torch.optim as optim

from models.common.trainer import Trainer

from .model import Tokenizer

class Trainer(Trainer):
    def __init__(self, args):
        self.args = args

        self.feat_funcs = args.get('feat_funcs', None)
        self.lang = args['lang'] # language determines how token normlization is done

    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._model = Tokenizer(self.args, self.args['vocab_size'], self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'])

            if self.args['mode'] == 'train':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
                self.optimizer = optim.Adam(self._model.parameters(), lr=self.args['lr0'], betas=(.9, .9), weight_decay=self.args['weight_decay'])

        return self._model

    def update(self, inputs):
        self.model.train()
        units, labels, features, _ = inputs

        if self.model.embeddings.weight.is_cuda:
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

        if self.model.embeddings.weight.is_cuda:
            units = units.cuda()
            labels = labels.cuda()
            features = features.cuda()

        pred = self.model(units, features)

        return pred.data.cpu().numpy()

    def save(self, filename):
        params = {
                'model': self.model.state_dict() if self.model is not None else None,
                'vocab': self.vocab,
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
            exit()
        self.args = checkpoint['config']
        self.model.load_state_dict(checkpoint['model'])
        self.vocab = checkpoint['vocab']
