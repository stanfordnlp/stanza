from copy import copy
import json
import numpy as np

from models.tokenize.data import TokenizerDataProcessor, TokenizerDataGenerator
from models.tokenize.trainer import TokenizerTrainer
from models.tokenize.vocab import Vocab

import utils.conll18_ud_eval as ud_eval

def load_mwt_dict(filename):
    if args['mwt_json_file'] is not None:
        with open(args['mwt_json_file'], 'r') as f:
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
            f.write("{}\t{}{}\t{}{}\n".format(i+1, tok, "\t_" * 4, i, "\t_" * 3))
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

def ud_scores(gold_conllu_file, system_conllu_file):
    gold_ud = ud_eval.load_conllu_file(gold_conllu_file)
    system_ud = ud_eval.load_conllu_file(system_conllu_file)
    evaluation = ud_eval.evaluate(gold_ud, system_ud)

    return evaluation

def harmonic_mean(a):
    if any([x == 0 for x in a]):
        return 0
    else:
        return len(a) / sum([1/x for x in a])

def eval_model(env):
    trainer = env.trainer

    oov_count, N = output_predictions(args['conll_file'], trainer, env.dev_data_generator, env.vocab, env.mwt_dict)
    scores = ud_scores(args['dev_conll_gold'], args['conll_file'])

    return harmonic_mean([scores['Tokens'].f1, scores['Sentences'].f1])

def train(env):
    args = env.args
    trainer = env.trainer
    N = len(env.data_generator)
    if args['cuda']:
        trainer.model.cuda()
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

def evaluate(env):
    args = env.args
    trainer = env.trainer
    trainer.load(args['save_name'])
    if args['cuda']:
        trainer.model.cuda()

    oov_count, N = output_predictions(args['conll_file'], trainer, env.data_generator, env.vocab, env.mwt_dict)

    print("OOV rate: {:6.3f}% ({:6d}/{:6d})".format(oov_count / N * 100, oov_count, N))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--json_file', type=str, default=None, help="JSON file with pre-chunked units")
    parser.add_argument('--mwt_json_file', type=str, default=None, help="JSON file for MWT expansions")
    parser.add_argument('--conll_file', type=str, default=None, help="CoNLL file for output")
    parser.add_argument('--vocab_file', type=str, default=None, help="Vocab file")
    parser.add_argument('--dev_txt_file', type=str, help="(Train only) Input plaintext file for the dev set")
    parser.add_argument('--dev_label_file', type=str, default=None, help="(Train only) Character-level label file for the dev set")
    parser.add_argument('--dev_json_file', type=str, default=None, help="(Train only) JSON file with pre-chunked units for the dev set")
    parser.add_argument('--dev_conll_gold', type=str, default=None, help="(Train only) CoNLL-U file for the dev set for early stopping")
    parser.add_argument('--lang', type=str, help="Language")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--emb_dim', type=int, default=30, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=100, help="Dimension of hidden units")
    parser.add_argument('--conv_filters', type=str, default="1,5,9,,1,5,9", help="Configuration of conv filters. ,, separates layers and , separates filter sizes in the same layer.")
    parser.add_argument('--residual', action='store_true', help="Add linear residual connections")
    parser.add_argument('--hierarchical', action='store_true', help="\"Hierarchical\" RNN tokenizer")
    parser.add_argument('--no-rnn', dest='rnn', action='store_false', help="Use CNN tokenizer")
    parser.add_argument('--input_dropout', action='store_true', help="Dropout input embeddings as well")
    parser.add_argument('--aux_clf', type=float, default=0.0, help="Strength for auxiliary classifiers; default 0 (don't use auxiliary classifiers)")
    parser.add_argument('--merge_aux_clf', action='store_true', help="Merge prediction from auxiliary classifiers with final classifier output")
    parser.add_argument('--conv_res', type=str, default=None, help="Convolutional residual layers for the RNN")
    parser.add_argument('--rnn_layers', type=int, default=1, help="Layers of RNN in the tokenizer")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm to clip to")
    parser.add_argument('--anneal', type=float, default=.9, help="Anneal the learning rate by this amount when dev performance deteriorate")
    parser.add_argument('--anneal_after', type=int, default=0, help="Anneal the learning rate no earlier than this step")
    parser.add_argument('--lr0', type=float, default=2e-3, help="Initial learning rate")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout probability")
    parser.add_argument('--unit_dropout', type=float, default=0.0, help="Unit dropout probability")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--max_seqlen', type=int, default=100, help="Maximum sequence length to consider at a time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use")
    parser.add_argument('--epochs', type=int, default=10, help="Total epochs to train the model for")
    parser.add_argument('--steps', type=int, default=None, help="Steps to train the model for, if unspecified use epochs")
    parser.add_argument('--report_steps', type=int, default=20, help="Update step interval to report loss")
    parser.add_argument('--shuffle_steps', type=int, default=0, help="Step interval to shuffle each paragragraph in the generator")
    parser.add_argument('--eval_steps', type=int, default=200, help="Step interval to evaluate the model on the dev set for early stopping")
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--save_dir', type=str, default='saved_models', help="Directory to save models in")
    parser.add_argument('--no_cuda', dest="cuda", action="store_false")

    args = parser.parse_args()

    args = vars(args)
    args['feat_funcs'] = ['space_before', 'capitalized', 'all_caps', 'numeric']
    args['feat_dim'] = len(args['feat_funcs'])
    args['save_name'] = "{}/{}".format(args['save_dir'], args['save_name']) if args['save_name'] is not None else '{}/{}_tokenizer.pkl'.format(args['save_dir'], args['lang'])

    env = Env(args)
    args['vocab_size'] = len(env.vocab)

    env.trainer = TokenizerTrainer(args)
    trainer = env.trainer

    env.mwt_dict = load_mwt_dict(args['mwt_json_file'])

    if args['mode'] == 'train':
        train(env)
    else:
        evaluate(env)
