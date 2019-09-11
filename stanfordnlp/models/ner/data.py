import random
import logging
import torch

from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common.vocab import PAD_ID, VOCAB_PREFIX
from stanfordnlp.models.pos.vocab import CharVocab, WordVocab
from stanfordnlp.models.ner.vocab import TagVocab, MultiVocab
from stanfordnlp.models.common.doc import *
from stanfordnlp.models.ner.utils import is_bio_scheme, convert_tags_to_bioes

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, doc, batch_size, args, pretrain=None, vocab=None, evaluation=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc

        data = self.load_doc(self.doc)
        self.tags = [[w[1] for w in sent] for sent in data]

        # handle vocab
        self.pretrain = pretrain
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            logger.debug("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, args)
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)
        logger.debug("{} batches created.".format(len(self.data)))

    def init_vocab(self, data):
        assert self.eval == False # for eval vocab must exist
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = self.pretrain.vocab
        tagvocab = TagVocab(data, self.args['shorthand'], idx=1)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'tag': tagvocab})
        return vocab

    def preprocess(self, data, vocab, args):
        processed = []
        if args['lowercase']: # handle word case
            case = lambda x: x.lower()
        else:
            case = lambda x: x
        for sent in data:
            processed_sent = [vocab['word'].map([case(w[0]) for w in sent])]
            processed_sent += [[vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [vocab['tag'].map([w[1] for w in sent])]
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
        assert len(batch) == 3

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

        tags = get_long_tensor(batch[2], batch_size)
        sentlens = [len(x) for x in batch[0]]
        return words, words_mask, wordchars, wordchars_mask, tags, orig_idx, word_orig_idx, sentlens, word_lens

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_doc(self, doc):
        data = doc.get([TEXT, NER], as_sentences=True)
        if not self.eval:
            data = self.process_tags(data)
        return data

    def process_tags(self, sentences):
        res = []
        # check if tag conversion is needed
        convert_to_bioes = False
        is_bio = is_bio_scheme([x[1] for sent in sentences for x in sent])
        if is_bio and self.args.get('scheme', 'bio').lower() == 'bioes':
            convert_to_bioes = True
            logger.debug("BIO tagging scheme found in input; converting into BIOES scheme...")
        # process tags
        for sent in sentences:
            words, tags = zip(*sent)
            # NER field sanity checking
            if self.eval and any([x is None or x == '_' for x in tags]):
                raise Exception("NER tag not found for some input data during training.")
            if convert_to_bioes:
                tags = convert_tags_to_bioes(tags)
            res.append([[w,t] for w,t in zip(words, tags)])
        return res

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        self.data = self.chunk_batches(data)
        random.shuffle(self.data)

    def chunk_batches(self, data):
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        return data

