import argparse
import json
import numpy as np
import random
import re
import torch

from bisect import bisect_right
from copy import copy
from stanfordnlp.models.common import conll
from stanfordnlp.models.common import utils
from stanfordnlp.models.tokenize.trainer import Trainer
from stanfordnlp.models.tokenize.vocab import Vocab

DEFAULT_TOKENIZE_CONFIG = {
    'mode': 'predict',
    'shorthand': 'en_ewt',
    'lang': 'en',
    'cuda': True,
    'max_seqlen': 1000,
    'feat_funcs': ['space_before', 'capitalized', 'all_caps', 'numeric'],
    'feat_dim': 4,
    'model_path': 'saved_models/tokenize/en_ewt_tokenizer.pt'
}

# class for loading data for the tokenizer
class TokenizeDataLoader:
    def __init__(self, text, config={}, vocab=None):
        # load arguments
        # look at submitted args to overwrite defaults
        self.args = DEFAULT_TOKENIZE_CONFIG
        for key in config.keys():
            self.args[key] = config[key]
        # set up labels
        labels = '\n\n'.join(['0' * len(pt.rstrip()) for pt in re.split('\n\s*\n', text)])
        # set up data
        self.data = [list(zip(re.sub('\s', ' ', pt.rstrip()), [int(x) for x in pc])) for pt, pc in
                     zip(re.split('\n\s*\n', text), labels.split('\n\n')) if len(pt.rstrip()) > 0]
        # set up vocab
        self.vocab = vocab if vocab is not None else self.init_vocab()
        # set up sentences
        self.sentences = [self.para_to_sentences(para) for para in self.data]
        self.init_sent_ids()
        # set up mwt_dict
        self.mwt_dict = None

    def init_vocab(self):
        vocab = Vocab(self.data, self.args['lang'])
        return vocab

    def init_sent_ids(self):
        self.sentence_ids = []
        self.cumlen = [0]
        for i, para in enumerate(self.sentences):
            for j in range(len(para)):
                self.sentence_ids += [(i, j)]
                self.cumlen += [self.cumlen[-1] + len(self.sentences[i][j])]

    def para_to_sentences(self, para):
        res = []
        funcs = []
        for feat_func in self.args['feat_funcs']:
            if feat_func == 'space_before':
                func = lambda x: 1 if x.startswith(' ') else 0
            elif feat_func == 'capitalized':
                func = lambda x: 1 if x[0].isupper() else 0
            elif feat_func == 'all_caps':
                func = lambda x: 1 if x.isupper() else 0
            elif feat_func == 'numeric':
                func = lambda x: 1 if (re.match('^([\d]+[,\.]*)+$', x) is not None) else 0
            else:
                assert False, 'Feature function "{}" is undefined.'.format(feat_func)

            funcs += [func]

        composite_func = lambda x: list(map(lambda f: f(x), funcs))

        def process_and_featurize(sent):
            return [(self.vocab.unit2id(y[0]), y[1], composite_func(y[0]), y[0]) for y in sent]

        current = []
        for unit, label in para:
            label1 = label if self.args['mode'] == 'train' else 0
            current += [[unit, label]]
            if label1 == 2 or label1 == 4: # end of sentence
                if len(current) <= self.args['max_seqlen']:
                    # get rid of sentences that are too long during training of the tokenizer
                    res += [process_and_featurize(current)]
                current = []

        if len(current) > 0:
            if self.args['mode'] == 'predict' or len(current) <= self.args['max_seqlen']:
                res += [process_and_featurize(current)]

        return res

    def __len__(self):
        return len(self.sentence_ids)

    def next(self, eval_offsets=None, unit_dropout=0.0):
        null_feats = [0] * len(self.sentences[0][0][0][2])
        def strings_starting(id_pair, offset=0, pad_len=self.args['max_seqlen']):
            pid, sid = id_pair
            res = copy(self.sentences[pid][sid][offset:])

            assert self.args['mode'] == 'predict' or len(res) <= self.args['max_seqlen'], 'The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it! {}'.format(self.args['max_seqlen'], len(res), ' '.join(["{}/{}".format(*x) for x in self.sentences[pid][sid]]))
            for sid1 in range(sid+1, len(self.sentences[pid])):
                res += self.sentences[pid][sid1]

                if self.args['mode'] != 'predict' and len(res) >= self.args['max_seqlen']:
                    res = res[:self.args['max_seqlen']]
                    break

            if unit_dropout > 0 and self.args['mode'] == 'train':
                unkid = self.vocab.unit2id('<UNK>')
                res = [(unkid, x[1], x[2], '<UNK>') if random.random() < unit_dropout else x for x in res]

            # pad with padding units and labels if necessary
            if pad_len > 0 and len(res) < pad_len:
                padid = self.vocab.unit2id('<PAD>')
                res += [(padid, -1, null_feats, '<PAD>')] * (pad_len - len(res))

            return res

        if eval_offsets is not None:
            # find max padding length
            pad_len = 0
            for eval_offset in eval_offsets:
                if eval_offset < self.cumlen[-1]:
                    pair_id = bisect_right(self.cumlen, eval_offset) - 1
                    pair = self.sentence_ids[pair_id]
                    pad_len = max(pad_len, len(strings_starting(pair, offset=eval_offset-self.cumlen[pair_id], pad_len=0)))

            res = []
            pad_len += 1
            for eval_offset in eval_offsets:
                # find unit
                if eval_offset >= self.cumlen[-1]:
                    padid = self.vocab.unit2id('<PAD>')
                    res += [[(padid, -1, null_feats, '<PAD>')] * pad_len]
                    continue

                pair_id = bisect_right(self.cumlen, eval_offset) - 1
                pair = self.sentence_ids[pair_id]
                res += [strings_starting(pair, offset=eval_offset-self.cumlen[pair_id], pad_len=pad_len)]
        else:
            id_pairs = random.sample(self.sentence_ids, min(len(self.sentence_ids), self.args['batch_size']))
            res = [strings_starting(pair) for pair in id_pairs]

        units = [[y[0] for y in x] for x in res]
        labels = [[y[1] for y in x] for x in res]
        features = [[y[2] for y in x] for x in res]
        raw_units = [[y[3] for y in x] for x in res]

        convert = lambda t: (torch.from_numpy(np.array(t[0], dtype=t[1])))

        units, labels, features = list(map(convert, [(units, np.int64), (labels, np.int64), (features, np.float32)]))

        return units, labels, features, raw_units


# class for running the tokenizer
class TokenizeProcessor:

    def __init__(self, config=DEFAULT_TOKENIZE_CONFIG):
        # set up configurations
        self.args = DEFAULT_TOKENIZE_CONFIG
        for key in config.keys():
            self.args[key] = config[key]
        # set up trainer
        self.trainer = Trainer(self.args)
        self.trainer.load(self.args['model_path'])
        # set configurations from loaded model
        loaded_args = self.trainer.args
        self.vocab = self.trainer.vocab
        self.args['vocab_size'] = len(self.vocab)
        for k in loaded_args:
            if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'save_name']:
                self.args[k] = loaded_args[k]
        if self.args['cuda']:
            self.trainer.model.cuda()
        # set up misc
        self.mwt_dict = None

    def build_input_data(self, text):
        return TokenizeDataLoader(text, self.args, vocab=self.vocab)

    def process(self, doc):
        return_sentences = []
        data_loader = self.build_input_data(doc.text)
        # more data set up for tokenization
        paragraphs = []
        for i, p in enumerate(data_loader.sentences):
            start = 0 if i == 0 else paragraphs[-1][2]
            length = sum([len(x) for x in p])
            paragraphs += [(i, start, start + length, length + 1)]  # para idx, start idx, end idx, length
        paragraphs = list(sorted(paragraphs, key=lambda x: x[3], reverse=True))
        all_preds = [None] * len(paragraphs)
        all_raw = [None] * len(paragraphs)
        eval_limit = max(3000, self.args['max_seqlen'])
        batch_size = self.trainer.args['batch_size']
        batches = int((len(paragraphs) + batch_size - 1) / batch_size)
        # process batches
        t = 0
        for i in range(batches):
            batchparas = paragraphs[i * batch_size: (i + 1) * batch_size]
            offsets = [x[1] for x in batchparas]
            t += sum([x[3] for x in batchparas])
            batch = data_loader.next(eval_offsets=offsets)
            raw = batch[3]
            N = len(batch[3][0])
            # get the predictions
            pred = np.argmax(self.trainer.predict(batch), axis=2)
            # post process after predictions
            for j, p in enumerate(batchparas):
                len1 = len([1 for x in raw[j] if x != '<PAD>'])
                if pred[j][len1 - 1] < 2:
                    pred[j][len1 - 1] = 2
                elif pred[j][len1 - 1] > 2:
                    pred[j][len1 - 1] = 4
                all_preds[p[0]] = pred[j][:len1]
                all_raw[p[0]] = raw[j]
        # generate output conll-u from predictions
        offset = 0
        oov_count = 0
        for j in range(len(paragraphs)):
            raw = all_raw[j]
            pred = all_preds[j]
            current_tok = ''
            current_sent = []
            for t, p in zip(raw, pred):
                if t == '<PAD>':
                    break
                # hack la_ittb
                if self.trainer.args['shorthand'] == 'la_ittb' and t in [":", ";"]:
                    p = 2
                offset += 1
                if self.vocab.unit2id(t) == self.vocab.unit2id('<UNK>'):
                    oov_count += 1
                current_tok += t
                if p >= 1:
                    tok = self.vocab.normalize_token(current_tok)
                    assert '\t' not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    current_sent += [(tok, p)]
                    current_tok = ''
                    if p == 2 or p == 4:
                        return_sentences.append(self.build_conllu(current_sent, self.mwt_dict))
                        current_sent = []
            if len(current_tok):
                tok = self.vocab.normalize_token(current_tok)
                assert '\t' not in tok, tok
                if len(tok) > 0:
                    current_sent += [(tok, 2)]
            if len(current_sent):
                return_sentences.append(self.build_conllu(current_sent, self.mwt_dict))
        # build initial conll_file
        conll_file = conll.CoNLLFile(input_str=('\n'.join(return_sentences)))
        # set doc's conll file
        doc.conll_file = conll_file

    def build_conllu(self, sentence, mwt_dict=None):
        return_string = ""
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
                return_string += ("{}-{}\t{}{}".format(i + 1, i + len(expansion), tok, "\t_" * 8))
                return_string += '\n'
                for etok in expansion:
                    return_string += ("{}\t{}{}\t{}{}".format(i + 1, etok, "\t_" * 4, i, "\t_" * 3))
                    return_string += '\n'
                    i += 1
            else:
                if len(tok) <= 0:
                    continue
                return_string += ("{}\t{}{}\t{}{}\t{}".format(
                    i + 1, tok, "\t_" * 4, i, "\t_" * 2, "MWT=Yes" if p == 3 or p == 4 else "_"))
                return_string += '\n'
                i += 1
        return return_string
