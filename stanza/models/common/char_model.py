from collections import Counter
from operator import itemgetter
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence, PackedSequence

from stanza.models.common.data import get_long_tensor
from stanza.models.common.packed_lstm import PackedLSTM
from stanza.models.common.utils import open_read_text, tensor_unsort, unsort
from stanza.models.common.dropout import SequenceUnitDropout
from stanza.models.common.vocab import UNK_ID, CharVocab

class CharacterModel(nn.Module):
    def __init__(self, args, vocab, pad=False, bidirectional=False, attention=True):
        super().__init__()
        self.args = args
        self.pad = pad
        self.num_dir = 2 if bidirectional else 1
        self.attn = attention

        # char embeddings
        self.char_emb = nn.Embedding(len(vocab['char']), self.args['char_emb_dim'], padding_idx=0)
        if self.attn: 
            self.char_attn = nn.Linear(self.num_dir * self.args['char_hidden_dim'], 1, bias=False)
            self.char_attn.weight.data.zero_()

        # modules
        self.charlstm = PackedLSTM(self.args['char_emb_dim'], self.args['char_hidden_dim'], self.args['char_num_layers'], batch_first=True, \
                dropout=0 if self.args['char_num_layers'] == 1 else args['dropout'], rec_dropout = self.args['char_rec_dropout'], bidirectional=bidirectional)
        self.charlstm_h_init = nn.Parameter(torch.zeros(self.num_dir * self.args['char_num_layers'], 1, self.args['char_hidden_dim']))
        self.charlstm_c_init = nn.Parameter(torch.zeros(self.num_dir * self.args['char_num_layers'], 1, self.args['char_hidden_dim']))

        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, chars, chars_mask, word_orig_idx, sentlens, wordlens):
        embs = self.dropout(self.char_emb(chars))
        batch_size = embs.size(0)
        embs = pack_padded_sequence(embs, wordlens, batch_first=True)
        output = self.charlstm(embs, wordlens, hx=(\
                self.charlstm_h_init.expand(self.num_dir * self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous(), \
                self.charlstm_c_init.expand(self.num_dir * self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous()))
         
        # apply attention, otherwise take final states
        if self.attn:
            char_reps = output[0]
            weights = torch.sigmoid(self.char_attn(self.dropout(char_reps.data)))
            char_reps = PackedSequence(char_reps.data * weights, char_reps.batch_sizes)
            char_reps, _ = pad_packed_sequence(char_reps, batch_first=True)
            res = char_reps.sum(1)
        else:
            h, c = output[1]
            res = h[-2:].transpose(0,1).contiguous().view(batch_size, -1)

        # recover character order and word separation
        res = tensor_unsort(res, word_orig_idx)
        res = pack_sequence(res.split(sentlens))
        if self.pad:
            res = pad_packed_sequence(res, batch_first=True)[0]

        return res

def build_charlm_vocab(path, cutoff=0):
    """
    Build a vocab for a CharacterLanguageModel

    Requires a large amount of memory, but only need to build once

    here we need some trick to deal with excessively large files
    for each file we accumulate the counter of characters, and
    at the end we simply pass a list of chars to the vocab builder
    """
    counter = Counter()
    if os.path.isdir(path):
        filenames = sorted(os.listdir(path))
    else:
        filenames = [os.path.split(path)[1]]
        path = os.path.split(path)[0]

    for filename in filenames:
        filename = os.path.join(path, filename)
        with open_read_text(filename) as fin:
            for line in fin:
                counter.update(list(line))

    if len(counter) == 0:
        raise ValueError("Training data was empty!")
    # remove infrequent characters from vocab
    for k in list(counter.keys()):
        if counter[k] < cutoff:
            del counter[k]
    # a singleton list of all characters
    data = [sorted([x[0] for x in counter.most_common()])]
    if len(data[0]) == 0:
        raise ValueError("All characters in the training data were less frequent than --cutoff!")
    vocab = CharVocab(data) # skip cutoff argument because this has been dealt with
    return vocab

class CharacterLanguageModel(nn.Module):

    def __init__(self, args, vocab, pad=False, is_forward_lm=True):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.is_forward_lm = is_forward_lm
        self.pad = pad
        self.finetune = True # always finetune unless otherwise specified

        # char embeddings
        self.char_emb = nn.Embedding(len(self.vocab['char']), self.args['char_emb_dim'], padding_idx=None) # we use space as padding, so padding_idx is not necessary
        
        # modules
        self.charlstm = PackedLSTM(self.args['char_emb_dim'], self.args['char_hidden_dim'], self.args['char_num_layers'], batch_first=True, \
                dropout=0 if self.args['char_num_layers'] == 1 else args['char_dropout'], rec_dropout = self.args['char_rec_dropout'], bidirectional=False)
        self.charlstm_h_init = nn.Parameter(torch.zeros(self.args['char_num_layers'], 1, self.args['char_hidden_dim']))
        self.charlstm_c_init = nn.Parameter(torch.zeros(self.args['char_num_layers'], 1, self.args['char_hidden_dim']))

        # decoder
        self.decoder = nn.Linear(self.args['char_hidden_dim'], len(self.vocab['char']))
        self.dropout = nn.Dropout(args['char_dropout'])
        self.char_dropout = SequenceUnitDropout(args.get('char_unit_dropout', 0), UNK_ID)

    def forward(self, chars, charlens, hidden=None):
        chars = self.char_dropout(chars)
        embs = self.dropout(self.char_emb(chars))
        batch_size = embs.size(0)
        embs = pack_padded_sequence(embs, charlens, batch_first=True)
        if hidden is None: 
            hidden = (self.charlstm_h_init.expand(self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous(),
                      self.charlstm_c_init.expand(self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous())
        output, hidden = self.charlstm(embs, charlens, hx=hidden)
        output = self.dropout(pad_packed_sequence(output, batch_first=True)[0])
        decoded = self.decoder(output)
        return output, hidden, decoded

    def get_representation(self, chars, charoffsets, charlens, char_orig_idx):
        with torch.no_grad():
            output, _, _ = self.forward(chars, charlens)
            res = [output[i, offsets] for i, offsets in enumerate(charoffsets)]
            res = unsort(res, char_orig_idx)
            res = pack_sequence(res)
            if self.pad:
                res = pad_packed_sequence(res, batch_first=True)[0]
        return res

    def build_char_representation(self, sentences):
        """
        Return values from this charlm for a list of list of words
        """
        CHARLM_START = "\n"
        CHARLM_END = " "

        forward = self.is_forward_lm
        vocab = self.char_vocab()
        device = next(self.parameters()).device

        all_data = []
        for idx, words in enumerate(sentences):
            if not forward:
                words = [x[::-1] for x in reversed(words)]

            chars = [CHARLM_START]
            offsets = []
            for w in words:
                chars.extend(w)
                chars.append(CHARLM_END)
                offsets.append(len(chars) - 1)
            if not forward:
                offsets.reverse()
            chars = vocab.map(chars)
            all_data.append((chars, offsets, len(chars), len(all_data)))

        all_data.sort(key=itemgetter(2), reverse=True)
        chars, char_offsets, char_lens, orig_idx = tuple(zip(*all_data))
        chars = get_long_tensor(chars, len(all_data), pad_id=vocab.unit2id(CHARLM_END)).to(device=device)

        with torch.no_grad():
            output, _, _ = self.forward(chars, char_lens)
            res = [output[i, offsets] for i, offsets in enumerate(char_offsets)]
            res = unsort(res, orig_idx)

        return res

    def hidden_dim(self):
        return self.args['char_hidden_dim']

    def char_vocab(self):
        return self.vocab['char']

    def train(self, mode=True):
        """
        Override the default train() function, so that when self.finetune == False, the training mode 
        won't be impacted by the parent models' status change.
        """
        if not mode: # eval() is always allowed, regardless of finetune status
            super().train(mode)
        else:
            if self.finetune: # only set to training mode in finetune status
                super().train(mode)

    def full_state(self):
        state = {
            'vocab': self.vocab['char'].state_dict(),
            'args': self.args,
            'state_dict': self.state_dict(),
            'pad': self.pad,
            'is_forward_lm': self.is_forward_lm
        }
        return state

    def save(self, filename):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        state = self.full_state()
        torch.save(state, filename, _use_new_zipfile_serialization=False)

    @classmethod
    def from_full_state(cls, state, finetune=False):
        vocab = {'char': CharVocab.load_state_dict(state['vocab'])}
        model = cls(state['args'], vocab, state['pad'], state['is_forward_lm'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.finetune = finetune # set finetune status
        return model

    @classmethod
    def load(cls, filename, finetune=False):
        state = torch.load(filename, lambda storage, loc: storage)
        # allow saving just the Model object,
        # and allow for old charlms to still work
        if 'state_dict' in state:
            return cls.from_full_state(state, finetune)
        return cls.from_full_state(state['model'], finetune)

class CharacterLanguageModelTrainer():
    def __init__(self, model, params, optimizer, criterion, scheduler, epoch=1, global_step=0):
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.epoch = epoch
        self.global_step = global_step

    def save(self, filename, full=True):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        state = {
            'model': self.model.full_state(),
            'epoch': self.epoch,
            'global_step': self.global_step,
        }
        if full and self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()
        if full and self.criterion is not None:
            state['criterion'] = self.criterion.state_dict()
        if full and self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        torch.save(state, filename, _use_new_zipfile_serialization=False)

    @classmethod
    def from_new_model(cls, args, vocab):
        model = CharacterLanguageModel(args, vocab, is_forward_lm=True if args['direction'] == 'forward' else False)
        if args['cuda']: model = model.cuda()
        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args['lr0'], momentum=args['momentum'], weight_decay=args['weight_decay'])
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=args['anneal'], patience=args['patience'])
        return cls(model, params, optimizer, criterion, scheduler)


    @classmethod
    def load(cls, args, filename, finetune=False):
        """
        Load the model along with any other saved state for training

        Note that you MUST set finetune=True if planning to continue training
        Otherwise the only benefit you will get will be a warm GPU
        """
        state = torch.load(filename, lambda storage, loc: storage)
        model = CharacterLanguageModel.from_full_state(state['model'], finetune)
        if args['cuda']: model = model.cuda()

        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args['lr0'], momentum=args['momentum'], weight_decay=args['weight_decay'])
        if 'optimizer' in state: optimizer.load_state_dict(state['optimizer'])

        criterion = torch.nn.CrossEntropyLoss()
        if 'criterion' in state: criterion.load_state_dict(state['criterion'])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=args['anneal'], patience=args['patience'])
        if 'scheduler' in state: scheduler.load_state_dict(state['scheduler'])

        epoch = state.get('epoch', 1)
        global_step = state.get('global_step', 0)
        return cls(model, params, optimizer, criterion, scheduler, epoch, global_step)

