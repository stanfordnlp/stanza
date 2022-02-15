import random
import logging
import torch

from stanza.models.common.bert_embedding import load_tokenizer
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanza.models.common.vocab import PAD_ID, VOCAB_PREFIX
from stanza.models.pos.vocab import CharVocab, WordVocab
from stanza.models.ner.vocab import TagVocab, MultiVocab
from stanza.models.common.doc import *
from stanza.models.ner.utils import process_tags

#tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)
#tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo-ner-dane")
#tokenizer = AutoTokenizer.from_pretrained("flax-community/roberta-base-danish")
#tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

logger = logging.getLogger('stanza')

class DataLoader:
    def __init__(self, doc, batch_size, args, pretrain=None, vocab=None, evaluation=False, preprocess_tags=True):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc
        self.preprocess_tags = preprocess_tags

        data = self.load_doc(self.doc)
        max_len_data = []
        self.tags = []

        #check if bert_model is used
        self.bert_tokenizer = load_tokenizer(self.args['bert_model']) if self.args.get('bert_model', False) else None
        
        if self.args.get('bert_model', False):
            #eliminate all the sentences that are too long for bert model
            for sent in data:
                #check if the max tokenized length is less than maximum (256 for vi) and replace nbs with space
                tokenized = [word[0].replace("\xa0","_").replace(" ", "_") for word in sent] if self.args['bert_model'].startswith("vinai/phobert") else [word[0].replace("\xa0"," ") for word in sent]
                #concatenate to a sentence
                sentence = ' '.join(tokenized)
                
                #tokenize using AutoTokenizer BERT
                tokenized = self.bert_tokenizer.tokenize(sentence)
                
                #convert tokens to ids
                sent_ids = self.bert_tokenizer.convert_tokens_to_ids(tokenized)
                
                #add start and end tokens to sent_ids
                tokenized_sent = [self.bert_tokenizer.bos_token_id] + sent_ids + [self.bert_tokenizer.eos_token_id]
                
                if len(tokenized_sent) > self.bert_tokenizer.model_max_length:
                    continue
                max_len_data.append(sent)
                self.tags.append([w[1] for w in sent])
            logger.info("Eliminated {} datapoints because their length is over maximum size of BERT model. ".format(len(data)-len(max_len_data)))
            data = max_len_data
        else:
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
        def from_model(model_filename):
            """ Try loading vocab from charLM model file. """
            state_dict = torch.load(model_filename, lambda storage, loc: storage)
            assert 'vocab' in state_dict, "Cannot find vocab in charLM model file."
            return state_dict['vocab']

        if self.eval:
            raise AssertionError("Vocab must exist for evaluation.")
        if self.args['charlm']:
            charvocab = CharVocab.load_state_dict(from_model(self.args['charlm_forward_file']))
        else: 
            charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = self.pretrain.vocab
        tagvocab = TagVocab(data, self.args['shorthand'], idx=1)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'tag': tagvocab})
        return vocab

    def preprocess(self, data, vocab, args):
        processed = []
        if args.get('lowercase', True): # handle word case
            case = lambda x: x.lower()
        else:
            case = lambda x: x
        if args.get('char_lowercase', False): # handle character case
            char_case = lambda x: x.lower()
        else:
            char_case = lambda x: x
        for sent in data:
            processed_sent = [[w[0] for w in sent]]
            processed_sent += [[vocab['char'].map([char_case(x) for x in w[0]]) for w in sent]]
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
        assert len(batch) == 3 # words: List[List[int]], chars: List[List[List[int]]], tags: List[List[int]]

        # sort sentences by lens for easy RNN operations
        sentlens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, sentlens)
        sentlens = [len(x) for x in batch[0]]

        # sort chars by lens for easy char-LM operations
        chars_forward, chars_backward, charoffsets_forward, charoffsets_backward, charlens = self.process_chars(batch[1])
        chars_sorted, char_orig_idx = sort_all([chars_forward, chars_backward, charoffsets_forward, charoffsets_backward], charlens)
        chars_forward, chars_backward, charoffsets_forward, charoffsets_backward = chars_sorted
        charlens = [len(sent) for sent in chars_forward]

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[1] for w in sent]
        wordlens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], wordlens)
        batch_words = batch_words[0]
        wordlens = [len(x) for x in batch_words]

        words = batch[0]
        
        wordchars = get_long_tensor(batch_words, len(wordlens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)
        chars_forward = get_long_tensor(chars_forward, batch_size, pad_id=self.vocab['char'].unit2id(' '))
        chars_backward = get_long_tensor(chars_backward, batch_size, pad_id=self.vocab['char'].unit2id(' '))
        chars = torch.cat([chars_forward.unsqueeze(0), chars_backward.unsqueeze(0)]) # padded forward and backward char idx
        charoffsets = [charoffsets_forward, charoffsets_backward] # idx for forward and backward lm to get word representation
        tags = get_long_tensor(batch[2], batch_size)

        return words, wordchars, wordchars_mask, chars, tags, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_doc(self, doc):
        data = doc.get([TEXT, NER], as_sentences=True, from_token=True)
        if self.preprocess_tags: # preprocess tags
            data = process_tags(data, self.args.get('scheme', 'bio'))
        return data

    def process_chars(self, sents):
        start_id, end_id = self.vocab['char'].unit2id('\n'), self.vocab['char'].unit2id(' ') # special token
        start_offset, end_offset = 1, 1
        chars_forward, chars_backward, charoffsets_forward, charoffsets_backward = [], [], [], []
        # get char representation for each sentence
        for sent in sents:
            chars_forward_sent, chars_backward_sent, charoffsets_forward_sent, charoffsets_backward_sent = [start_id], [start_id], [], []
            # forward lm
            for word in sent:
                chars_forward_sent += word
                charoffsets_forward_sent = charoffsets_forward_sent + [len(chars_forward_sent)] # add each token offset in the last for forward lm
                chars_forward_sent += [end_id]
            # backward lm
            for word in sent[::-1]:
                chars_backward_sent += word[::-1]
                charoffsets_backward_sent = [len(chars_backward_sent)] + charoffsets_backward_sent # add each offset in the first for backward lm
                chars_backward_sent += [end_id]
            # store each sentence
            chars_forward.append(chars_forward_sent)
            chars_backward.append(chars_backward_sent)
            charoffsets_forward.append(charoffsets_forward_sent)
            charoffsets_backward.append(charoffsets_backward_sent)
        charlens = [len(sent) for sent in chars_forward] # forward lm and backward lm should have the same lengths
        return chars_forward, chars_backward, charoffsets_forward, charoffsets_backward, charlens

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(data)
        self.data = self.chunk_batches(data)

    def chunk_batches(self, data):
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        return data

