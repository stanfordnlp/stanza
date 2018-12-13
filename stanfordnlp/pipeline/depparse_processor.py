import random
import torch

from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import conll
from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.common.vocab import PAD_ID, VOCAB_PREFIX, ROOT_ID, CompositeVocab
from stanfordnlp.models.depparse.trainer import Trainer
from stanfordnlp.models.pos.vocab import CharVocab, WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanfordnlp.models.pos.xpos_vocab_factory import xpos_vocab_factory

DEFAULT_DEPPARSE_CONFIG = {
    'data_dir': 'data/depparse', 'wordvec_dir': 'extern_data/word2vec', 'train_file': None,
    'eval_file': 'parser_input.conllu',
    'output_file': 'parser_output.conllu',
    'gold_file': 'parser_input.conllu',
    'pretrain_path': 'saved_models/depparse/en_ewt_parser.pretrain.pt',
    'model_path': 'saved_models/depparse/en_ewt_parser.pt',
    'mode': 'predict',
    'lang': 'en',
    'shorthand': 'en_ewt',
    'best_param': False,
    'hidden_dim': 400,
    'char_hidden_dim': 400,
    'deep_biaff_hidden_dim': 400,
    'composite_deep_biaff_hidden_dim': 100,
    'word_emb_dim': 75,
    'char_emb_dim': 100,
    'tag_emb_dim': 50,
    'transformed_dim': 125,
    'num_layers': 3,
    'char_num_layers': 1,
    'word_dropout': 0.33,
    'dropout': 0.5,
    'rec_dropout': 0,
    'char_rec_dropout': 0,
    'char': True,
    'pretrain': True,
    'linearization': True,
    'distance': True,
    'sample_train': 1.0,
    'optim': 'adam',
    'lr': 0.003,
    'beta2': 0.95,
    'max_steps': 50000,
    'eval_interval': 100,
    'max_steps_before_stop': 3000,
    'batch_size': 5000,
    'max_grad_norm': 1.0,
    'log_step': 20,
    'save_dir': 'saved_models/depparse',
    'save_name': None,
    'seed': 1234,
    'cuda': True,
    'cpu': False
}

class DepparseDataLoader:
    def __init__(self, doc, batch_size, args, pretrain, vocab=None, evaluation=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        self.conll, data = self.load_data(doc)

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab
        self.pretrain_vocab = pretrain.vocab

        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)

    def init_vocab(self, data):
        assert self.eval == False # for eval vocab must exist
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=7, lower=True)
        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(data, self.args['shorthand'])
        featsvocab = FeatureVocab(data, self.args['shorthand'], idx=3)
        lemmavocab = WordVocab(data, self.args['shorthand'], cutoff=7, idx=4, lower=True)
        deprelvocab = WordVocab(data, self.args['shorthand'], idx=6)
        vocab = MultiVocab({'char': charvocab,
                'word': wordvocab,
                'upos': uposvocab,
                'xpos': xposvocab,
                'feats': featsvocab,
                'lemma': lemmavocab,
                'deprel': deprelvocab})
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab, args):
        processed = []
        xpos_replacement = [[ROOT_ID] * len(vocab['xpos'])] if isinstance(vocab['xpos'], CompositeVocab) else [ROOT_ID]
        feats_replacement = [[ROOT_ID] * len(vocab['feats'])]
        for sent in data:
            processed_sent = [[ROOT_ID] + vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[[ROOT_ID]] + [vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [[ROOT_ID] + vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [xpos_replacement + vocab['xpos'].map([w[2] for w in sent])]
            processed_sent += [feats_replacement + vocab['feats'].map([w[3] for w in sent])]
            processed_sent += [[ROOT_ID] + pretrain_vocab.map([w[0] for w in sent])]
            processed_sent += [[ROOT_ID] + vocab['lemma'].map([w[4] for w in sent])]
            processed_sent += [[int(w[5]) for w in sent]]
            processed_sent += [vocab['deprel'].map([w[6] for w in sent])]
            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 9

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]
        word_lens = [len(x) for x in batch_words]

        # convert to tensors
        words = batch[0]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        upos = get_long_tensor(batch[2], batch_size)
        xpos = get_long_tensor(batch[3], batch_size)
        ufeats = get_long_tensor(batch[4], batch_size)
        pretrained = get_long_tensor(batch[5], batch_size)
        sentlens = [len(x) for x in batch[0]]
        lemma = get_long_tensor(batch[6], batch_size)
        head = get_long_tensor(batch[7], batch_size)
        deprel = get_long_tensor(batch[8], batch_size)
        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, orig_idx, word_orig_idx, sentlens, word_lens

    def load_data(self, doc):
        data = doc.conll_file.get(['word', 'upos', 'xpos', 'feats', 'lemma', 'head', 'deprel'], as_sentences=True)
        return doc.conll_file, data

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(data)
        self.data = self.chunk_batches(data)

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key = lambda x: len(x[0]) + random.random() * 5)

        current = []
        currentlen = 0
        for x in data:
            if len(x[0]) + currentlen > self.batch_size:
                res.append(current)
                current = []
                currentlen = 0
            current.append(x)
            currentlen += len(x[0])

        if currentlen > 0:
            res.append(current)

        return res


class DepparseProcessor:

    def __init__(self, config={}):
        # set up configurations
        self.args = DEFAULT_DEPPARSE_CONFIG
        for key in config.keys():
            self.args[key] = config[key]
        # get pretrained word vectors
        self.pretrain = Pretrain(self.args['pretrain_path'])
        # set up trainer
        self.trainer = Trainer(pretrain=self.pretrain, model_file=self.args['model_path'])
        self.loaded_args, self.vocab = self.trainer.args, self.trainer.vocab
        for k in self.args:
            if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
                self.loaded_args[k] = self.args[k]
        self.loaded_args['cuda'] = self.args['cuda'] and not self.args['cpu']

    def process(self, doc):
        batch = DepparseDataLoader(
            doc, self.loaded_args['batch_size'], self.loaded_args, self.pretrain, vocab=self.vocab, evaluation=True)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        batch.conll.set(['head', 'deprel'], [y for x in preds for y in x])
        return batch.conll
