import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from .data import TokenizerDataGenerator
from .tokenizer_models import Tokenizer, RNNTokenizer
from .vocab import Vocab

class TokenizerTrainer:
    def __init__(self, args):
        self.args = args
        if args['json_file'] is not None:
            with open(args['json_file']) as f:
                self.data = json.load(f)
        else:
            with open(args['txt_file']) as f:
                text = ''.join(f.readlines()).rstrip()

            if args['label_file'] is not None:
                with open(args['label_file']) as f:
                    labels = ''.join(f.readlines()).rstrip()
            else:
                labels = '\n\n'.join(['0' * len(pt) for pt in text.split('\n\n')])

            self.data = [list(zip(pt.rstrip(), [int(x) for x in pc])) for pt, pc in zip(text.split('\n\n'), labels.split('\n\n'))]

        self.data_generator = TokenizerDataGenerator(args, self.data)
        self.feat_funcs = args.get('feat_funcs', None)
        self.lang = args['lang'] # language determines how token normlization is done

    @property
    def vocab(self):
        # enable lazy construction in case we're just loading the vocab from file
        if not hasattr(self, '_vocab'):
            self._vocab = Vocab(self.data, self.lang)

        return self._vocab

    @property
    def model(self):
        if not hasattr(self, '_model'):
            if self.args['rnn']:
                self._model = RNNTokenizer(self.args, len(self.vocab), self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'])
            else:
                self._model = Tokenizer(self.args, len(self.vocab), self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'])

            if self.args['mode'] == 'train':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
                self.opt = optim.Adam(self._model.parameters(), lr=2e-3, betas=(.9, .9), weight_decay=self.args['weight_decay'])

        return self._model

    def update(self, inputs):
        self.model.train()
        units, labels, features, _ = inputs

        if self.model.embeddings.weight.is_cuda:
            units = units.cuda()
            labels = labels.cuda()
            features = features.cuda()

        pred, aux_outputs = self.model(units, features)

        self.opt.zero_grad()
        classes = pred.size(2)
        loss = self.criterion(pred.view(-1, classes), labels.view(-1))

        if self.args['aux_clf'] > 0 and not self.args['merge_aux_clf']:
            for aux_output in aux_outputs:
                loss += self.args['aux_clf'] * self.criterion(aux_output.view(-1, classes), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.opt.step()

        return loss.item()

    def change_lr(self, new_lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = new_lr

    def predict(self, inputs):
        self.model.eval()
        units, labels, features, _ = inputs

        if self.model.embeddings.weight.is_cuda:
            units = units.cuda()
            labels = labels.cuda()
            features = features.cuda()

        pred, _ = self.model(units, features)

        return pred.data.cpu().numpy()

    def save(self, filename):
        savedict = {
                   'vocab': self.vocab,
                   'model': self.model.state_dict(),
                   'optim': self.opt.state_dict()
                   }
        with open(filename, 'wb') as f:
            pickle.dump(savedict, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            savedict = pickle.load(f)

        self._vocab = savedict['vocab']
        self.model.load_state_dict(savedict['model'])
        if self.args['mode'] == 'train':
            self.opt.load_state_dict(savedict['optim'])
