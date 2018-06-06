from bisect import bisect_left
from collections import Counter
from copy import copy
import json
import pickle
import numpy as np
import random
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils.postprocess_vietnamese_tokenizer_data import paras_to_chunks

class Tokenizer(nn.Module):
    def __init__(self, nchars, emb_dim, hidden_dim, N_CLASSES=4, dropout=0):
        super().__init__()

        self.args = args
        feat_dim = args['feat_dim']

        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)

        self.conv_sizes = [[int(y) for y in x.split(',')] for x in args['conv_filters'].split(',,')]

        self.conv_filters = nn.ModuleList()

        if args['residual']:
            self.res_filters = nn.ModuleList()

        if args['aux_clf'] > 0 or args['merge_aux_clf']:
            self.aux_clf = nn.ModuleList()

        for layer, sizes in enumerate(self.conv_sizes):
            thislayer = nn.ModuleList()
            in_dim = emb_dim + feat_dim if layer == 0 else len(self.conv_sizes[layer-1]) * hidden_dim
            for size in sizes:
                thislayer.append(nn.Conv1d(in_dim, hidden_dim, size, padding=size//2))

            self.conv_filters.append(thislayer)

            if args['residual']:
                self.res_filters.append(nn.Conv1d(in_dim, len(self.conv_sizes[layer]) * hidden_dim, 1))

            if (args['aux_clf'] > 0 or args['merge_aux_clf']) and layer < len(self.conv_sizes) - 1:
                self.aux_clf.append(nn.Conv1d(hidden_dim * len(self.conv_sizes[layer]), N_CLASSES, 1))

        self.dense_clf = nn.Conv1d(hidden_dim * len(self.conv_sizes[-1]), N_CLASSES, 1)

        self.dropout = nn.Dropout(dropout)
        self.input_dropout = nn.Dropout(dropout)

    def forward(self, x, feats):
        emb = self.embeddings(x)

        if self.args['input_dropout']:
            emb = self.input_dropout(emb)

        emb = torch.cat([emb, feats], 2)

        emb = emb.transpose(1, 2).contiguous()
        inp = emb

        aux_outputs = []

        for layeri, layer in enumerate(self.conv_filters):
            out = [f(inp) for f in layer]
            out = torch.cat(out, 1)
            out = F.relu(out)
            if self.args['residual']:
                out += self.res_filters[layeri](inp)
            if (self.args['aux_clf'] > 0 or self.args['merge_aux_clf']) and layeri < len(self.conv_sizes) - 1:
                aux_outputs += [self.aux_clf[layeri](out).transpose(1, 2).contiguous()]
            if layeri < len(self.conv_filters) - 1:
                out = self.dropout(out)
            inp = out

        pred = self.dense_clf(inp)
        pred = pred.transpose(1, 2).contiguous()

        if self.args['merge_aux_clf']:
            for aux_out in aux_outputs:
                pred += aux_out
            pred /= len(self.conv_sizes)

        return pred, aux_outputs

class RNNTokenizer(nn.Module):
    def __init__(self, nchars, emb_dim, hidden_dim, N_CLASSES=4, dropout=0):
        super().__init__()

        self.args = args
        feat_dim = args['feat_dim']

        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)

        self.rnn = nn.LSTM(emb_dim + feat_dim, hidden_dim, num_layers=self.args['rnn_layers'], bidirectional=True, batch_first=True, dropout=dropout if self.args['rnn_layers'] > 1 else 0)
        #self.rnn2 = nn.LSTM(emb_dim + feat_dim + 1, hidden_dim, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        if self.args['conv_res'] is not None:
            self.conv_res = nn.ModuleList()
            self.conv_sizes = [int(x) for x in self.args['conv_res'].split(',')]

            for size in self.conv_sizes:
                self.conv_res.append(nn.Conv1d(emb_dim + feat_dim, hidden_dim * 2, size, padding=size//2))

        self.dense_clf = nn.Linear(hidden_dim * 2, N_CLASSES)
        self.dense_clf2 = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, feats):
        emb = self.embeddings(x)

        emb = self.dropout(emb)

        emb = torch.cat([emb, feats], 2)

        inp, _ = self.rnn(emb)

        if self.args['conv_res'] is not None:
            conv_input = emb.transpose(1, 2).contiguous()
            for l in self.conv_res:
                inp = inp + l(conv_input).transpose(1, 2).contiguous()

        inp = self.dropout(inp)

        pred0 = self.dense_clf(inp)

        pred0_ = F.log_softmax(pred0, 2)

        #emb = torch.cat([emb, pred0_[:,:,0].unsqueeze(2)], 2)
        inp2, _ = self.rnn2(inp * (1 - torch.exp(pred0_[:,:,0].unsqueeze(2))))
        inp2 = self.dropout(inp2)

        pred1 = self.dense_clf2(inp2)

        pred = torch.cat([pred0[:,:,:1], pred0[:,:,2].unsqueeze(2) + pred1, pred0[:,:,3].unsqueeze(2)], 2)

        return pred, []

class Vocab:
    def __init__(self, paras, lang):
        self.lang = lang
        self.build_vocab(paras)

    def build_vocab(self, paras):
        counter = Counter()
        for para in paras:
            for unit in para:
                normalized = self.normalize_unit(unit[0])
                counter[normalized] += 1

        self._id2unit = ['<PAD>', '<UNK>'] + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

    def unit2id(self, unit):
        unit = self.normalize_unit(unit)
        if unit in self._unit2id:
            return self._unit2id[unit]
        else:
            return self._unit2id['<UNK>']

    def id2unit(self, id):
        return self._id2unit[id]

    def normalize_unit(self, unit):
        # Normalize minimal units used by the tokenizer
        # For Vietnamese this means a syllable, for other languages this means a character
        normalized = unit
        if self.lang == 'vi':
            normalized = normalized.lstrip()

        return normalized

    def normalize_token(self, token):
        token = token.lstrip().replace('\n', ' ')

        if self.lang in ['zh', 'ja', 'ko']:
            token = token.replace(' ', '')

        return token

    def __len__(self):
        return len(self._id2unit)

class TokenizerDataGenerator:
    def __init__(self, args, data):
        self.args = args

        # data comes in a list of paragraphs, where each paragraph is a list of units with unit-level labels
        self.sentences = [self.para_to_sentences(para) for para in data]

        self.init_sent_ids()

    def init_sent_ids(self):
        self.sentence_ids = []
        self.cumlen = [0]
        for i, para in enumerate(self.sentences):
            for j in range(len(para)):
                self.sentence_ids += [(i, j)]
                self.cumlen += [self.cumlen[-1] + len(self.sentences[i][j])]

    def para_to_sentences(self, para):
        res = []

        current = []
        for unit, label in para:
            current += [[unit, label]]
            if label == 2: # end of sentence
                if len(current) <= self.args['max_seqlen']:
                    # get rid of sentences that are too long during training of the tokenizer
                    res += [current]
                current = []

        if len(current) > 0:
            if args['mode'] == 'predict' or len(current) <= self.args['max_seqlen']:
                res += [current]

        return res

    def __len__(self):
        return len(self.sentence_ids)

    def shuffle(self):
        for para in self.sentences:
            random.shuffle(para)
        self.init_sent_ids()

    def next(self, vocab, feat_funcs=['space_before', 'capitalized'], eval_offset=-1, unit_dropout=0.0):
        def strings_starting(id_pair, offset=0):
            pid, sid = id_pair
            res = copy(self.sentences[pid][sid][offset:])

            assert args['mode'] == 'predict' or len(res) <= args['max_seqlen'], 'The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it! {}'.format(args['max_seqlen'], len(res), ' '.join(["{}/{}".format(*x) for x in self.sentences[pid][sid]]))
            for sid1 in range(sid+1, len(self.sentences[pid])):
                res += self.sentences[pid][sid1]

                if args['mode'] != 'predict' and len(res) >= args['max_seqlen']:
                    res = res[:args['max_seqlen']]
                    break

            if unit_dropout > 0:
                res = [('<UNK>', x[1]) if random.random() < unit_dropout else x for x in res]

            # pad with padding units and labels if necessary
            if len(res) < args['max_seqlen']:
                res += [('<PAD>', -1)] * (args['max_seqlen'] - len(res))

            return res

        if eval_offset >= 0:
            # find unit
            if eval_offset >= self.cumlen[-1]:
                return None

            pair_id = bisect_left(self.cumlen, eval_offset)
            pair = self.sentence_ids[pair_id]
            res = [strings_starting(pair, offset=eval_offset-self.cumlen[pair_id])]
        else:
            id_pairs = random.sample(self.sentence_ids, self.args['batch_size'])
            res = [strings_starting(pair) for pair in id_pairs]

        funcs = []
        for feat_func in feat_funcs:
            if feat_func == 'space_before':
                func = lambda x: x.startswith(' ')
            elif feat_func == 'capitalized':
                func = lambda x: x[0].isupper()
            elif feat_func == 'all_caps':
                func = lambda x: x.isupper()
            elif feat_func == 'numeric':
                func = lambda x: re.match('^[\d]+$', x) is not None
            else:
                assert False, 'Feature function "{}" is undefined.'.format(feat_func)

            funcs += [func]

        composite_func = lambda x: list(map(lambda f: f(x), funcs))

        features = [[composite_func(y[0]) for y in x] for x in res]

        units = [[vocab.unit2id(y[0]) for y in x] for x in res]
        raw_units = [[y[0] for y in x] for x in res]
        labels = [[y[1] for y in x] for x in res]

        convert = lambda t: (torch.from_numpy(np.array(t[0], dtype=t[1])))

        units, labels, features = list(map(convert, [(units, np.int64), (labels, np.int64), (features, np.float32)]))

        return units, labels, features, raw_units

class TokenizerTrainer:
    def __init__(self, args):
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
        self.args = args
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
                self._model = RNNTokenizer(len(self.vocab), args['emb_dim'], args['hidden_dim'], dropout=args['dropout'])
            else:
                self._model = Tokenizer(len(self.vocab), args['emb_dim'], args['hidden_dim'], dropout=args['dropout'])

            if args['mode'] == 'train':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
                self.opt = optim.Adam(self._model.parameters(), lr=2e-3, betas=(.9, .9), weight_decay=args['weight_decay'])

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--json_file', type=str, default=None, help="JSON file with pre-chunked units")
    parser.add_argument('--mwt_json_file', type=str, default=None, help="JSON file for MWT expansions")
    parser.add_argument('--conll_file', type=str, default=None, help="CoNLL file for output")
    parser.add_argument('--lang', type=str, help="Language")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--emb_dim', type=int, default=30, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=100, help="Dimension of hidden units")
    parser.add_argument('--conv_filters', type=str, default="1,5,9,,1,5,9", help="Configuration of conv filters. ,, separates layers and , separates filter sizes in the same layer.")
    parser.add_argument('--residual', action='store_true', help="Add linear residual connections")
    parser.add_argument('--no-rnn', dest='rnn', action='store_false', help="Use CNN tokenizer")
    parser.add_argument('--input_dropout', action='store_true', help="Dropout input embeddings as well")
    parser.add_argument('--aux_clf', type=float, default=0.0, help="Strength for auxiliary classifiers; default 0 (don't use auxiliary classifiers)")
    parser.add_argument('--merge_aux_clf', action='store_true', help="Merge prediction from auxiliary classifiers with final classifier output")
    parser.add_argument('--conv_res', type=str, default=None, help="Convolutional residual layers for the RNN")
    parser.add_argument('--rnn_layers', type=int, default=1, help="Layers of RNN in the tokenizer")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm to clip to")
    parser.add_argument('--anneal', type=float, default=0, help="(Equivalent) frequency to half the learning rate; 0 means no annealing (the default)")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout probability")
    parser.add_argument('--unit_dropout', type=float, default=0.0, help="Unit dropout probability")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--max_seqlen', type=int, default=100, help="Maximum sequence length to consider at a time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use")
    parser.add_argument('--epochs', type=int, default=10, help="Total epochs to train the model for")
    parser.add_argument('--steps', type=int, default=None, help="Steps to train the model for, if unspecified use epochs")
    parser.add_argument('--report_steps', type=int, default=20, help="Update step interval to report loss")
    parser.add_argument('--shuffle_steps', type=int, default=0, help="Step interval to shuffle each paragragraph in the generator")
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--save_dir', type=str, default='saved_models', help="Directory to save models in")
    parser.add_argument('--no_cuda', dest="cuda", action="store_false")

    args = parser.parse_args()

    args = vars(args)
    args['feat_funcs'] = ['space_before', 'capitalized', 'all_caps', 'numeric']
    args['feat_dim'] = len(args['feat_funcs'])
    args['save_name'] = "{}/{}".format(args['save_dir'], args['save_name']) if args['save_name'] is not None else '{}/{}_tokenizer.pkl'.format(args['save_dir'], args['lang'])
    trainer = TokenizerTrainer(args)

    N = len(trainer.data_generator)
    if args['mode'] == 'train':
        if args['cuda']:
            trainer.model.cuda()
        steps = args['steps'] if args['steps'] is not None else int(N * args['epochs'] / args['batch_size'] + .5)
        lr0 = 2e-3

        for step in range(steps):
            batch = trainer.data_generator.next(trainer.vocab, feat_funcs=trainer.feat_funcs, unit_dropout=args['unit_dropout'])

            loss = trainer.update(batch)
            if step % args['report_steps'] == 0:
                print("Step {:6d}/{:6d} Loss: {:.3f}".format(step, steps, loss))

            if args['shuffle_steps'] > 0 and step % args['shuffle_steps'] == 0:
                trainer.data_generator.shuffle()

            if args['anneal'] > 0:
                trainer.change_lr(lr0 * (.5 ** ((step + 1) / args['anneal'])))

        trainer.save(args['save_name'])
    else:
        trainer.load(args['save_name'])
        if args['cuda']:
            trainer.model.cuda()

        offset = 0
        oov_count = 0

        mwt_dict = None
        if args['mwt_json_file'] is not None:
            with open(args['mwt_json_file'], 'r') as f:
                mwt_dict0 = json.load(f)

            mwt_dict = dict()
            for item in mwt_dict0:
                (key, expansion), count = item

                if key not in mwt_dict or mwt_dict[key][1] < count:
                    mwt_dict[key] = (expansion, count)

        def print_sentence(sentence, f, mwt_dict=None):
            i = 0
            for tok, p in current_sent:
                expansion = None
                if p == 3 and mwt_dict is not None:
                    # MWT found, (attempt to) expand it!
                    if tok in mwt_dict:
                        expansion = mwt_dict[tok][0]
                    elif tok.lower() in mwt_dict:
                        expansion = mwt_dict[tok.lower()][0]
                if expansion is not None:
                    f.write("{}-{}\t{}{}\n".format(i+1, i+len(expansion), tok, "\t_" * 8))
                    for etok in expansion:
                        f.write("{}\t{}{}\t{}{}\n".format(i+1, etok, "\t_" * 4, i, "\t_" * 3))
                        i += 1
                else:
                    if len(tok) <= 0:
                        continue
                    f.write("{}\t{}{}\t{}{}\n".format(i+1, tok, "\t_" * 4, i, "\t_" * 3))
                    i += 1
            f.write('\n')

        with open(args['conll_file'], 'w') as f:
            while True:
                batch = trainer.data_generator.next(trainer.vocab, feat_funcs=trainer.feat_funcs, eval_offset=offset)
                if batch is None:
                    break
                pred = np.argmax(trainer.predict(batch)[0], axis=1)

                current_tok = ''
                current_sent = []

                for t, p in zip(batch[3][0], pred):
                    if t == '<PAD>':
                        break
                    offset += 1
                    if trainer.vocab.unit2id(t) == trainer.vocab.unit2id('<UNK>'):
                        oov_count += 1

                    current_tok += t
                    if p >= 1:
                        current_sent += [(trainer.vocab.normalize_token(current_tok), p)]
                        current_tok = ''
                        if p == 2:
                            print_sentence(current_sent, f, mwt_dict)
                            current_sent = []

                if len(current_tok):
                    current_sent += [(trainer.vocab.normalize_token(current_tok), 2)]

                if len(current_sent):
                    print_sentence(current_sent, f, mwt_dict)

        print("OOV rate: {:6.3f}% ({:6d}/{:6d})".format(oov_count / offset * 100, oov_count, offset))
