from bisect import bisect_left
from copy import copy
import numpy as np
import random
from sru import SRU
import torch
import torch.nn as nn
import torch.optim as optim
from .tokenizer import Vocab, TokenizerTrainer

class SentenceSplitter(nn.Module):
    def __init__(self, nwords, emb_dim, hidden_dim, N_CLASSES=2, dropout=0):
        super().__init__()

        self.embeddings = nn.Embedding(nwords, emb_dim, padding_idx=0)

        #self.rnn = SRU(emb_dim, hidden_dim, num_layers=1, dropout=dropout, use_tanh=1, bidirectional=True)
        #self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=1, dropout=dropout, bidirectional=True)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=1, dropout=dropout, bidirectional=True)
        self.clf = nn.Linear(hidden_dim * 2, N_CLASSES)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embeddings(x.transpose(0, 1))

        emb = self.dropout(emb)

        hid, _ = self.rnn(emb)

        hid = self.dropout(hid)

        pred = self.clf(hid)
        pred = pred.transpose(0, 1).contiguous()

        return pred

class SSplitDataGenerator:
    def __init__(self, args, conll_file):
        self.args = args
        self.sentences = self.conll2sentences(conll_file)
        self.init_cumlen()

    def init_cumlen(self):
        self.cumlen = [0]
        for sent in self.sentences:
            self.cumlen += [self.cumlen[-1] + len(sent)]

    def shuffle(self):
        random.shuffle(self.sentences)
        self.init_cumlen()

    def conll2sentences(self, conll_file):
        res = []
        cur = []
        with open(conll_file) as f:
            for line in f:
                line = line.strip()

                if line.startswith('#'):
                    continue

                if len(line) <= 0:
                    if len(cur) > 0:
                        cur[-1][1] = 1
                        res += [cur]
                        cur = []
                    continue

                line = line.split('\t')
                cur += [[line[1], 0]]

        if len(cur) > 0:
            cur[-1][1] = 1
            res += [cur]

        return res

    def __len__(self):
        return len(self.sentences)

    def next(self, vocab, eval_offset=-1, unit_dropout=0.0):
        def strings_starting(sid, offset=0):
            res = copy(self.sentences[sid][offset:])

            assert args['mode'] == 'predict' or len(res) <= args['max_seqlen'], 'The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it! {}'.format(args['max_seqlen'], len(res), ' '.join(["{}/{}".format(*x) for x in self.sentences[sid]]))
            for sid1 in range(sid+1, len(self.sentences)):
                res += self.sentences[sid1]

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

            sid = bisect_left(self.cumlen, eval_offset)
            res = [strings_starting(sid, offset=eval_offset-self.cumlen[sid])]
        else:
            sids = random.sample(range(len(self.sentences)), self.args['batch_size'])
            res = [strings_starting(sid) for sid in sids]

        units = [[vocab.unit2id(y[0]) for y in x] for x in res]
        raw_units = [[y[0] for y in x] for x in res]
        labels = [[y[1] for y in x] for x in res]

        convert = lambda t: (torch.from_numpy(np.array(t[0], dtype=t[1])))

        units, labels = list(map(convert, [(units, np.int64), (labels, np.int64)]))

        return units, labels, raw_units

class SSplitTrainer(TokenizerTrainer):
    def __init__(self, args):
        self.data_generator = SSplitDataGenerator(args, args['in_conll_file'])
        self.args = args
        self.lang = args['lang']

    @property
    def vocab(self):
        # enable lazy construction in case we're just loading the vocab from file
        if not hasattr(self, '_vocab'):
            self._vocab = Vocab(self.data_generator.sentences, self.lang)

        return self._vocab

    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._model = SentenceSplitter(len(self.vocab), args['emb_dim'], args['hidden_dim'], dropout=args['dropout'])

            if args['mode'] == 'train':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
                self.opt = optim.Adam(self._model.parameters(), lr=2e-3, betas=(.9, .9), weight_decay=args['weight_decay'])

        return self._model

    def update(self, inputs):
        self.model.train()
        units, labels, _ = inputs

        if self.model.embeddings.weight.is_cuda:
            units = units.cuda()
            labels = labels.cuda()

        pred = self.model(units)

        self.opt.zero_grad()
        classes = pred.size(2)
        loss = self.criterion(pred.view(-1, classes), labels.view(-1))

        if self.args['aux_clf'] > 0:
            for aux_output in aux_outputs:
                loss += self.args['aux_clf'] * self.criterion(aux_output.view(-1, classes), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.opt.step()

        return loss.item()

    def predict(self, inputs):
        self.model.eval()
        units, labels, _ = inputs

        if self.model.embeddings.weight.is_cuda:
            units = units.cuda()
            labels = labels.cuda()

        pred = self.model(units)

        return pred.data.cpu().numpy()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--json_file', type=str, default=None, help="JSON file with pre-chunked units")
    parser.add_argument('--mwt_json_file', type=str, default=None, help="JSON file for MWT expansions")
    parser.add_argument('--in_conll_file', type=str, default=None, help="CoNLL file for input")
    parser.add_argument('--conll_file', type=str, default=None, help="CoNLL file for output")
    parser.add_argument('--lang', type=str, help="Language")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--emb_dim', type=int, default=30, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=200, help="Dimension of hidden units")
    parser.add_argument('--conv_filters', type=str, default="1,5,9,,1,5,9", help="Configuration of conv filters. ,, separates layers and , separates filter sizes in the same layer.")
    parser.add_argument('--residual', action='store_true', help="Add linear residual connections")
    parser.add_argument('--aux_clf', type=float, default=0.0, help="Strength for auxiliary classifiers; default 0 (don't use auxiliary classifiers)")

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
    args['save_name'] = "{}/{}".format(args['save_dir'], args['save_name']) if args['save_name'] is not None else '{}/{}_tokenizer.pkl'.format(args['save_dir'], args['lang'])
    trainer = SSplitTrainer(args)

    N = len(trainer.data_generator)
    if args['mode'] == 'train':
        if args['cuda']:
            trainer.model.cuda()
        steps = args['steps'] if args['steps'] is not None else int(N * args['epochs'] / args['batch_size'] + .5)
        lr0 = 2e-3

        for step in range(steps):
            batch = trainer.data_generator.next(trainer.vocab, unit_dropout=args['unit_dropout'])

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
                batch = trainer.data_generator.next(trainer.vocab, eval_offset=offset)
                if batch is None:
                    break
                pred = np.argmax(trainer.predict(batch)[0], axis=1)

                current_tok = ''
                current_sent = []

                for t, p in zip(batch[2][0], pred):
                    if t == '<PAD>':
                        break
                    offset += 1
                    if trainer.vocab.unit2id(t) == trainer.vocab.unit2id('<UNK>'):
                        oov_count += 1

                    current_tok += t
                    if p >= 0:
                        current_sent += [(trainer.vocab.normalize_token(current_tok), p)]
                        current_tok = ''
                        if p == 1:
                            print_sentence(current_sent, f, mwt_dict)
                            current_sent = []

                if len(current_tok):
                    current_sent += [(trainer.vocab.normalize_token(current_tok), 2)]

                if len(current_sent):
                    print_sentence(current_sent, f, mwt_dict)

        print("OOV rate: {:6.3f}% ({:6d}/{:6d})".format(oov_count / offset * 100, oov_count, offset))
