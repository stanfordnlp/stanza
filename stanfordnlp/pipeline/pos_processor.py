import random
import torch

from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import conll
from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.common.vocab import PAD_ID, VOCAB_PREFIX
from stanfordnlp.models.pos.trainer import Trainer
from stanfordnlp.models.pos.vocab import CharVocab, WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanfordnlp.models.pos.xpos_vocab_factory import xpos_vocab_factory

DEFAULT_POS_CONFIG = {
                      'data_dir': 'data/pos',
                      'wordvec_dir': 'extern_data/word2vec',
                      'train_file': None,
                      'eval_file': 'pre_pos_content.conllu',
                      'output_file': 'post_pos_content.conllu',
                      'gold_file': 'pre_pos_content.conllu',
                      'pretrain_path': 'saved_models/pos/en_ewt_tagger.pretrain.pt',
                      'model_path': 'saved_models/pos/en_ewt_tagger.pt',
                      'mode': 'predict',
                      'lang': 'en_ewt',
                      'shorthand': 'en_ewt',
                      'best_param': False,
                      #'hidden_dim': 200,
                      #'char_hidden_dim': 400,
                      #'deep_biaff_hidden_dim': 400,
                      #'composite_deep_biaff_hidden_dim': 100,
                      #'word_emb_dim': 75,
                      #'char_emb_dim': 100,
                      #'tag_emb_dim': 50,
                      #'transformed_dim': 125,
                      #'num_layers': 2,
                      #'char_num_layers': 1,
                      #'word_dropout': 0.33,
                      #'dropout': 0.5,
                      #'rec_dropout': 0,
                      #'char_rec_dropout': 0,
                      'char': True,
                      'pretrain': True,
                      'share_hid': False,
                      'sample_train': 1.0,
                      'optim': 'adam',
                      'lr': 0.003,
                      'beta2': 0.95,
                      'max_steps': 50000,
                      'eval_interval': 100,
                      'adapt_eval_interval': True,
                      'max_steps_before_stop': 3000,
                      'batch_size': 5000,
                      'max_grad_norm': 1.0,
                      'log_step': 20,
                      'save_dir': 'saved_models/pos',
                      'save_name': None,
                      'seed': 1234,
                      'cuda': True,
                      'cpu': False
                      }

class POSDataLoader:
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

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)

    def init_vocab(self, data):
        assert self.eval == False  # for eval vocab must exist
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=7, lower=True)
        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(data, self.args['shorthand'])
        featsvocab = FeatureVocab(data, self.args['shorthand'], idx=3)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab,
                            'xpos': xposvocab,
                            'feats': featsvocab})
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab, args):
        processed = []
        for sent in data:
            processed_sent = [vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [vocab['xpos'].map([w[2] for w in sent])]
            processed_sent += [vocab['feats'].map([w[3] for w in sent])]
            processed_sent += [pretrain_vocab.map([w[0] for w in sent])]
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
        assert len(batch) == 6

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
        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, orig_idx, word_orig_idx, sentlens, word_lens

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_data(self, doc):
        data = doc.conll_file.get(['word', 'upos', 'xpos', 'feats'], as_sentences=True)
        return doc.conll_file, data

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(data)
        self.data = self.chunk_batches(data)

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key=lambda x: len(x[0]) + random.random() * 5)

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

class POSProcessor:

    def __init__(self, config={}):
        # set up configurations
        self.args = DEFAULT_POS_CONFIG
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
        batch = POSDataLoader(
            doc, self.loaded_args['batch_size'], self.loaded_args, self.pretrain, vocab=self.vocab, evaluation=True)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        batch.conll.set(['upos', 'xpos', 'feats'], [y for x in preds for y in x])

    def write_conll(self, batch):
        """ Write current conll contents to file.
        """
        return_string = ""
        for sent in batch.conll.sents:
            for ln in sent:
                return_string += ("\t".join(ln))
                return_string += "\n"
            return_string += "\n"
        return return_string

