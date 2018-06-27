from collections import Counter
from copy import copy
import json
import numpy as np

from models.common.utils import ud_scores, harmonic_mean, load_config, save_config

from .data import TokenizerDataProcessor, TokenizerDataGenerator
from .trainer import TokenizerTrainer
from .vocab import Vocab

def load_mwt_dict(filename):
    if filename is not None:
        with open(filename, 'r') as f:
            mwt_dict0 = json.load(f)

        mwt_dict = dict()
        for item in mwt_dict0:
            (key, expansion), count = item

            if key not in mwt_dict or mwt_dict[key][1] < count:
                mwt_dict[key] = (expansion, count)

        return mwt_dict
    else:
        return

def print_sentence(sentence, f, mwt_dict=None):
    i = 0
    for tok, p in sentence:
        expansion = None
        if (p == 3 or p == 4) and mwt_dict is not None:
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
            f.write("{}\t{}{}\t{}{}\t{}\n".format(i+1, tok, "\t_" * 4, i, "\t_" * 2, "MWT=Yes" if p == 3 or p == 4 else "_"))
            i += 1
    f.write('\n')

def output_predictions(output_filename, trainer, data_generator, vocab, mwt_dict, max_seqlen=1000):
    paragraphs = []
    for i, p in enumerate(data_generator.sentences):
        start = 0 if i == 0 else paragraphs[-1][2]
        length = sum([len(x) for x in p])
        paragraphs += [(i, start, start+length, length)] # para idx, start idx, end idx, length

    paragraphs = list(sorted(paragraphs, key=lambda x: x[3], reverse=True))

    all_preds = [None] * len(paragraphs)
    all_raw = [None] * len(paragraphs)

    eval_limit = max(3000, max_seqlen)

    batch_size = trainer.args['batch_size']
    batches = int((len(paragraphs) + batch_size - 1) / batch_size)

    t = 0
    for i in range(batches):
        batchparas = paragraphs[i * batch_size : (i + 1) * batch_size]
        offsets = [x[1] for x in batchparas]
        t += sum([x[3] for x in batchparas])

        batch = data_generator.next(eval_offsets=offsets)
        raw = batch[3]

        N = len(batch[3][0])
        if N <= eval_limit:
            pred = np.argmax(trainer.predict(batch), axis=2)
        else:
            idx = [0] * len(batchparas)
            Ns = [p[3] for p in batchparas]
            pred = [[] for _ in batchparas]
            while True:
                ens = [min(N - idx1, eval_limit) for idx1, N in zip(idx, Ns)]
                en = max(ens)
                batch1 = batch[0][:, :en], batch[1][:, :en], batch[2][:, :en], [x[:en] for x in batch[3]]
                pred1 = np.argmax(trainer.predict(batch1), axis=2)

                for j in range(len(batchparas)):
                    sentbreaks = np.where((pred1[j] == 2) + (pred1[j] == 4))[0]
                    if len(sentbreaks) <= 0 or idx[j] >= Ns[j] - eval_limit:
                        advance = ens[j]
                    else:
                        advance = np.max(sentbreaks) + 1

                    pred[j] += [pred1[j, :advance]]
                    idx[j] += advance

                if all([idx1 >= N for idx1, N in zip(idx, Ns)]):
                    break
                batch = data_generator.next(eval_offsets=[x+y for x, y in zip(idx, offsets)])

            pred = [np.concatenate(p, 0) for p in pred]

        for j, p in enumerate(batchparas):
            len1 = len([1 for x in raw[j] if x != '<PAD>'])
            all_preds[p[0]] = pred[j][:len1]
            all_raw[p[0]] = raw[j]

    offset = 0
    oov_count = 0
    with open(output_filename, 'w') as f:
        for j in range(len(paragraphs)):
            raw = all_raw[j]
            pred = all_preds[j]

            current_tok = ''
            current_sent = []

            for t, p in zip(raw, pred):
                if t == '<PAD>':
                    break
                # hack la_ittb
                if trainer.args['shorthand'] == 'la_ittb' and t in [":", ";"]:
                    p = 2
                offset += 1
                if vocab.unit2id(t) == vocab.unit2id('<UNK>'):
                    oov_count += 1

                current_tok += t
                if p >= 1:
                    tok = vocab.normalize_token(current_tok)
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    current_sent += [(tok, p)]
                    current_tok = ''
                    if p == 2 or p == 4:
                        print_sentence(current_sent, f, mwt_dict)
                        current_sent = []

            if len(current_tok):
                tok = vocab.normalize_token(current_tok)
                if len(tok) > 0:
                    current_sent += [(tok, 2)]

            if len(current_sent):
                print_sentence(current_sent, f, mwt_dict)

    return oov_count, offset, all_preds

class Env:
    def __init__(self, args):
        self.args = args

    @property
    def data_processor(self):
        if not hasattr(self, '_data_processor'):
            self._data_processor = TokenizerDataProcessor(self.args['json_file'], self.args['txt_file'], self.args['label_file'])
        return self._data_processor

    @property
    def dev_data_processor(self):
        if not hasattr(self, '_dev_data_processor'):
            self._dev_data_processor = TokenizerDataProcessor(self.args['dev_json_file'], self.args['dev_txt_file'], self.args['dev_label_file'])
        return self._dev_data_processor

    @property
    def data_generator(self):
        if not hasattr(self, '_data_generator'):
            self._data_generator = TokenizerDataGenerator(self.args, self.vocab, self.data_processor.data)
        return self._data_generator

    @property
    def dev_data_generator(self):
        if not hasattr(self, '_dev_data_generator'):
            args1 = copy(self.args)
            args1['mode'] = 'predict'
            self._dev_data_generator = TokenizerDataGenerator(args1, self.vocab, self.dev_data_processor.data)
        return self._dev_data_generator

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self._vocab = Vocab(self.args['vocab_file'], self.data_processor.data, self.args['lang'])
        return self._vocab


def eval_model(env):
    trainer = env.trainer
    args = env.args

    oov_count, N, all_preds = output_predictions(args['conll_file'], trainer, env.dev_data_generator, env.vocab, env.mwt_dict, args['max_seqlen'])

    scores = ud_scores(args['dev_conll_gold'], args['conll_file'])

    print(args['shorthand'], scores['Tokens'].f1, scores['Sentences'].f1, scores['Words'].f1)
#    return harmonic_mean([scores['Words'].f1, scores['Sentences'].f1], [.5, 1])
    all_preds = np.concatenate(all_preds, 0)
    labels = [y[1] for x in env.dev_data_processor.data for y in x]
    counter = Counter(zip(all_preds, labels))

    def f1(pred, gold, mapping):
        pred = [mapping[p] for p in pred]
        gold = [mapping[g] for g in gold]

        lastp = -1; lastg = -1
        tp = 0; fp = 0; fn = 0
        for i, (p, g) in enumerate(zip(pred, gold)):
            if p == g > 0 and lastp == lastg:
                lastp = i
                lastg = i
                tp += 1
            elif p > 0 and g > 0:
                lastp = i
                lastg = i
                fp += 1
                fn += 1
            elif p > 0:
                # and g == 0
                lastp = i
                fp += 1
            elif g > 0:
                lastg = i
                fn += 1

        if tp == 0:
            return 0
        else:
            return 2 * tp / (2 * tp + fp + fn)

    f1tok = f1(all_preds, labels, {0:0, 1:1, 2:1, 3:1, 4:1})
    f1sent = f1(all_preds, labels, {0:0, 1:0, 2:1, 3:0, 4:1})
    f1mwt = f1(all_preds, labels, {0:0, 1:1, 2:1, 3:2, 4:2})
    print(args['shorthand'], f1tok, f1sent, f1mwt)
    return harmonic_mean([f1tok, f1sent, f1mwt], [1, 1, .01])

def train(env):
    args = env.args
    save_config(args, '{}/{}_config.json'.format(args['save_dir'], args['shorthand']))
    trainer = TokenizerTrainer(args)
    env.trainer = trainer
    if args['cuda']:
        trainer.model.cuda()

    if args['load_name'] is not None:
        load_name = "{}/{}".format(args['save_dir'], args['load_name'])
        trainer.load(load_name)
    trainer.change_lr(args['lr0'])

    N = len(env.data_generator)
    steps = args['steps'] if args['steps'] is not None else int(N * args['epochs'] / args['batch_size'] + .5)
    lr = args['lr0']

    prev_dev_score = -1
    best_dev_score = -1
    best_dev_step = -1

    for step in range(1, steps+1):
        batch = env.data_generator.next(unit_dropout=args['unit_dropout'])

        loss = trainer.update(batch)
        if step % args['report_steps'] == 0:
            print("Step {:6d}/{:6d} Loss: {:.3f}".format(step, steps, loss))

        if args['shuffle_steps'] > 0 and step % args['shuffle_steps'] == 0:
            env.data_generator.shuffle()

        if step % args['eval_steps'] == 0:
            dev_score = eval_model(env)
            reports = ['Dev score: {:6.3f}'.format(dev_score * 100)]
            if step >= args['anneal_after'] and dev_score < prev_dev_score:
                reports += ['lr: {:.6f} -> {:.6f}'.format(lr, lr * args['anneal'])]
                lr *= args['anneal']
                trainer.change_lr(lr)

            prev_dev_score = dev_score

            if dev_score > best_dev_score:
                reports += ['New best dev score!']
                best_dev_score = dev_score
                best_dev_step = step
                trainer.save(args['save_name'])
            print('\t'.join(reports))

    print('Best dev score={} at step {}'.format(best_dev_score, best_dev_step))

    env.param_manager.update(args, best_dev_score)

def evaluate(env):
    args = env.args
    config_file = '{}/{}_config.json'.format(args['save_dir'], args['shorthand'])
    loaded_args = load_config(config_file)
    for k in loaded_args:
        if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'save_name']:
            args[k] = loaded_args[k]
    trainer = TokenizerTrainer(args)
    trainer.load(args['save_name'])
    if args['cuda']:
        trainer.model.cuda()

    oov_count, N, _ = output_predictions(args['conll_file'], trainer, env.data_generator, env.vocab, env.mwt_dict, args['max_seqlen'])

    print("OOV rate: {:6.3f}% ({:6d}/{:6d})".format(oov_count / N * 100, oov_count, N))
