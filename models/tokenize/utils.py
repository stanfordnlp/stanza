from copy import copy
import json
import numpy as np

from models.common.utils import ud_scores, harmonic_mean, load_config

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
            f.write("{}\t{}{}\t{}{}\t{}\n".format(i+1, tok, "\t_" * 4, i, "\t_" * 2, "MWT=Yes" if p == 3 else "_"))
            i += 1
    f.write('\n')

def output_predictions(output_filename, trainer, data_generator, vocab, mwt_dict):
    offset = 0
    oov_count = 0

    with open(output_filename, 'w') as f:
        while True:
            batch = data_generator.next(vocab, feat_funcs=trainer.feat_funcs, eval_offset=offset)
            if batch is None:
                break
            pred = np.argmax(trainer.predict(batch)[0], axis=1)

            current_tok = ''
            current_sent = []

            for t, p in zip(batch[3][0], pred):
                if t == '<PAD>':
                    break
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
                    if p == 2:
                        print_sentence(current_sent, f, mwt_dict)
                        current_sent = []

            if len(current_tok):
                tok = vocab.normalize_token(current_tok)
                if len(tok) > 0:
                    current_sent += [(tok, 2)]

            if len(current_sent):
                print_sentence(current_sent, f, mwt_dict)

    return oov_count, offset

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
            self._data_generator = TokenizerDataGenerator(self.args, self.data_processor.data)
        return self._data_generator

    @property
    def dev_data_generator(self):
        if not hasattr(self, '_dev_data_generator'):
            args1 = copy(self.args)
            args1['mode'] = 'predict'
            self._dev_data_generator = TokenizerDataGenerator(args1, self.dev_data_processor.data)
        return self._dev_data_generator

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self._vocab = Vocab(self.args['vocab_file'], self.data_processor.data, self.args['lang'])
        return self._vocab


def eval_model(env):
    trainer = env.trainer
    args = env.args

    oov_count, N = output_predictions(args['conll_file'], trainer, env.dev_data_generator, env.vocab, env.mwt_dict)
    scores = ud_scores(args['dev_conll_gold'], args['conll_file'])

    return harmonic_mean([scores['Words'].f1, scores['Sentences'].f1])

def train(env):
    args = env.args
    trainer = env.trainer
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

    for step in range(1, steps+1):
        batch = env.data_generator.next(env.vocab, feat_funcs=trainer.feat_funcs, unit_dropout=args['unit_dropout'])

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
                trainer.save(args['save_name'])
            print('\t'.join(reports))

    env.param_manager.update(args, best_dev_score)

def evaluate(env):
    args = env.args
    config_file = '{}/{}_config.json'.format(args['save_dir'], args['shorthand'])
    loaded_args = load_config(config_file)
    for k in loaded_args:
        if not k.endswith('_file') and k not in ['cuda']:
            args[k] = loaded_args[k]
    trainer = env.trainer
    trainer.load(args['save_name'])
    if args['cuda']:
        trainer.model.cuda()

    oov_count, N = output_predictions(args['conll_file'], trainer, env.data_generator, env.vocab, env.mwt_dict)

    print("OOV rate: {:6.3f}% ({:6d}/{:6d})".format(oov_count / N * 100, oov_count, N))
