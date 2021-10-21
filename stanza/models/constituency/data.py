import random
import logging
import torch
import sys
#from itertools.groupby
import math
from pympler import asizeof
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizerFast
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanza.models.common.vocab import PAD_ID, VOCAB_PREFIX
from stanza.models.pos.vocab import CharVocab, WordVocab
from stanza.models.ner.vocab import TagVocab, MultiVocab
from stanza.models.common.doc import *
from stanza.models.ner.utils import process_tags

#phobert = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(torch.device("cuda:0"))
#tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
phobert = AutoModel.from_pretrained("vinai/phobert-base").to(torch.device("cuda:0"))
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)

#phobert = AutoModel.from_pretrained("xlm-roberta-base").to(torch.device("cuda:0"))
#tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=False)
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
        new_data = []
        for sent in data:
            new_sent = []
            for word in sent:
                word = (word[0].replace("‑","‐"), word[1])
                if word[0] in (".", "…","...", "?", "!", ";"):
                    word = (word[0].replace("…","..."),word[1])
                    if new_sent:
                        new_sent.append(word)
                        new_data.append(new_sent)
                        new_sent = []
                    else:
                        new_sent.append(word)
                        
                else:
                    new_sent.append(word)
            if new_sent:
                new_data.append(new_sent)
                new_sent = []
                
        #data = new_data[:16000]
        data = new_data
        print(max([len(sent) for sent in data]))
        print([ sent for sent in data if len(sent) > 256])
        #print(data)
            #i = (list(g) for _, g in groupby(sentence, key='.'.__ne__))
            #print([a + b for a, b in zip_longest(i, i, fillvalue=[])])


        count = 0
        
        data = [sent for sent in data if len(sent)<=150]
        
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
        tokenized_sents = []
        print("start checking bert emb...")
        print("length of data: ", len(data))

        list_tokenized = []
        for sent in data:
            #replace \xa0 or whatever the space character is by _ since PhoBERT expects _ between syllables
            tokenized = [word[0].replace("\xa0","_") for word in sent]

            #concatenate to a sentence
            sentence = ' '.join(tokenized)

            #tokenize using AutoTokenizer PhoBERT
            tokenized = tokenizer.tokenize(sentence)

            #add tokenized to list_tokenzied for later checking
            list_tokenized.append(tokenized)

            #convert tokens to ids
            sent_ids = tokenizer.convert_tokens_to_ids(tokenized)
            #add start and end tokens to sent_ids
            tokenized_sent = [0] + sent_ids + [2]
        
            #tokenized_sent = [word[0].replace("\xa0"," ") for word in sent]
            
            if len(tokenized_sent)>256:
                print(len(tokenized_sent))
                print("oops", tokenized_sent)

            #add to tokenized_sents
            tokenized_sents.append(torch.tensor(tokenized_sent).detach())
            #processed_sent = [vocab['word'].map([case(w[0]) for w in sent])]
            processed_sent = [[vocab['char'].map([char_case(x) for x in w[0]]) for w in sent]]
            processed_sent += [vocab['tag'].map([w[1] for w in sent])]
            processed.append(processed_sent)

            #print("done loading bert emb!")


        
        size = len(tokenized_sents)

        #padding the inputs
        tokenized_sents_padded = torch.nn.utils.rnn.pad_sequence(tokenized_sents,batch_first=True,padding_value=1)
        
        features = []

        #Feed into PhoBERT 128 at a time.
        for i in range(int(math.ceil(size/128))):
            with torch.no_grad():
                
                feature = phobert(torch.tensor(tokenized_sents_padded[128*i:128*i+128]).to(torch.device("cuda:0")), output_hidden_states=True)
            #take the second output layer since experiments shows it give the best result
            features += torch.tensor(feature[2][-2]).detach().cpu()
            del feature
            
            print("done ", (i+1)*128)
            print(asizeof.asizeof(features))
            #print(len(features))
        print(len(processed))
        #print(len(tokenized))

        print("Length of features", len(features))
        assert len(features)==size
        assert len(features)==len(processed)

        #process the output
        for idx, sent in enumerate(processed):
            #only take the vector of the last word piece of a word/ you can do other methods such as first word piece or averaging.
            new_sent=[features[idx][idx2 +1].numpy() for idx2, i in enumerate(list_tokenized[idx]) if (idx2 > 0  and not list_tokenized[idx][idx2-1].endswith("@@")) or (idx2==0)]
            #add new vector to processed
            processed[idx] = [new_sent] + processed[idx]
            
            if len(processed[idx][0]) != len(processed[idx][1]):
                print(len(processed[idx][0]), len(processed[idx][1]))
                #print(processed[idx])
                print(list_tokenized[idx])
            assert len(processed[idx][0]) == len(processed[idx][1])
            #processed[idx] = [features[idx][1:1+end_list[idx]]] + processed[idx]
        #del list_tokenized
        del tokenized_sents
        del tokenized
        del features
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
        #print("START")
        #print(batch)
        #print(batch_size)
        #print("END")
        #print(len(batch))
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

        # convert to tensors
        words = get_float_tensor(batch[0], batch_size)
        #print(words)
        words_mask = torch.sum(torch.abs(words),2) == 0.0
        #words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(wordlens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)
        chars_forward = get_long_tensor(chars_forward, batch_size, pad_id=self.vocab['char'].unit2id(' '))
        chars_backward = get_long_tensor(chars_backward, batch_size, pad_id=self.vocab['char'].unit2id(' '))
        chars = torch.cat([chars_forward.unsqueeze(0), chars_backward.unsqueeze(0)]) # padded forward and backward char idx
        charoffsets = [charoffsets_forward, charoffsets_backward] # idx for forward and backward lm to get word representation
        tags = get_long_tensor(batch[2], batch_size)

        return words, words_mask, wordchars, wordchars_mask, chars, tags, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets

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

