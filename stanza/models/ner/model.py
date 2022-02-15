import enum
import math
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizerFast, AutoModelForPreTraining, AutoModelForMaskedLM

from stanza.models.common.packed_lstm import PackedLSTM
from stanza.models.common.dropout import WordDropout, LockedDropout
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from stanza.models.common.crf import CRFLoss
from stanza.models.common.vocab import PAD_ID
from stanza.models.common.bert_embedding import extract_phobert_embeddings, extract_bert_embeddings
logger = logging.getLogger('stanza')

class NERTagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, bert_model=None, bert_tokenizer=None, use_cuda=False):
        super().__init__()

        self.use_cuda = use_cuda
        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")

        def add_unsaved_module(name, module):    
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            self.word_emb = nn.Embedding(len(self.vocab['word']), self.args['word_emb_dim'], PAD_ID)
            # load pretrained embeddings if specified
            if emb_matrix is not None:
                self.init_emb(emb_matrix)
            if not self.args.get('emb_finetune', True):
                self.word_emb.weight.detach_()
            input_size += self.args['word_emb_dim']
            if self.args.get('bert_model', False):
                input_size += self.bert_model.config.hidden_size

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args['charlm']:
                if args['charlm_forward_file'] is None or not os.path.exists(args['charlm_forward_file']):
                    raise FileNotFoundError('Could not find forward character model: {}  Please specify with --charlm_forward_file'.format(args['charlm_forward_file']))
                if args['charlm_backward_file'] is None or not os.path.exists(args['charlm_backward_file']):
                    raise FileNotFoundError('Could not find backward character model: {}  Please specify with --charlm_backward_file'.format(args['charlm_backward_file']))
                add_unsaved_module('charmodel_forward', CharacterLanguageModel.load(args['charlm_forward_file'], finetune=False))
                add_unsaved_module('charmodel_backward', CharacterLanguageModel.load(args['charlm_backward_file'], finetune=False))
                input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
            else:
                self.charmodel = CharacterModel(args, vocab, bidirectional=True, attention=False)
                input_size += self.args['char_hidden_dim'] * 2

        # optionally add a input transformation layer
        if self.args.get('input_transform', False):
            self.input_transform = nn.Linear(input_size, input_size)
        else:
            self.input_transform = None
       
        # recurrent layers
        self.taggerlstm = PackedLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, \
                bidirectional=True, dropout=0 if self.args['num_layers'] == 1 else self.args['dropout'])
        # self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.drop_replacement = None
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']), requires_grad=False)
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']), requires_grad=False)

        # tag classifier
        num_tag = len(self.vocab['tag'])
        self.tag_clf = nn.Linear(self.args['hidden_dim']*2, num_tag)
        self.tag_clf.bias.data.zero_()

        # criterion
        self.crit = CRFLoss(num_tag)

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])
        self.lockeddrop = LockedDropout(args['locked_dropout'])

    def init_emb(self, emb_matrix):
        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = torch.from_numpy(emb_matrix)
        vocab_size = len(self.vocab['word'])
        dim = self.args['word_emb_dim']
        assert emb_matrix.size() == (vocab_size, dim), \
            "Input embedding matrix must match size: {} x {}, found {}".format(vocab_size, dim, emb_matrix.size())
        self.word_emb.weight.data.copy_(emb_matrix)

    def forward(self, sentences, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx):
        
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        
        inputs = []
        batch_size = len(sentences)

        #extract static embeddings
        static_words, word_mask = self.extract_static_embeddings(self.args, sentences)

        if self.use_cuda:
            word_mask = word_mask.cuda()
            static_words = static_words.cuda()

        if self.args.get('bert_model', False):
            processed_bert = extract_bert_embeddings(self.args['bert_model'], self.bert_tokenizer, self.bert_model, sentences, self.device)
            bert_words = get_float_tensor(processed_bert, len(processed_bert))
            assert(bert_words[0].size(0)==tags[0].size(0))
            if self.use_cuda:
                bert_words = bert_words.cuda()

        if self.args['word_emb_dim'] > 0:
            word_static_emb = self.word_emb(static_words)
            word_emb = pack(torch.cat((word_static_emb, torch.tensor(bert_words)), 2)) if self.args.get('bert_model', False) else pack(word_static_emb)
            inputs += [word_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                char_reps_forward = self.charmodel_forward.get_representation(chars[0], charoffsets[0], charlens, char_orig_idx)
                char_reps_forward = PackedSequence(char_reps_forward.data, char_reps_forward.batch_sizes)
                char_reps_backward = self.charmodel_backward.get_representation(chars[1], charoffsets[1], charlens, char_orig_idx)
                char_reps_backward = PackedSequence(char_reps_backward.data, char_reps_backward.batch_sizes)
                inputs += [char_reps_forward, char_reps_backward]
            else:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(char_reps.data, char_reps.batch_sizes)
                inputs += [char_reps]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        if self.args['word_dropout'] > 0:
            lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = pad(lstm_inputs)
        lstm_inputs = self.lockeddrop(lstm_inputs)
        lstm_inputs = pack(lstm_inputs).data

        if self.input_transform:
            lstm_inputs = self.input_transform(lstm_inputs)

        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(\
                self.taggerlstm_h_init.expand(2 * self.args['num_layers'], batch_size, self.args['hidden_dim']).contiguous(), \
                self.taggerlstm_c_init.expand(2 * self.args['num_layers'], batch_size, self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data


        # prediction layer
        lstm_outputs = self.drop(lstm_outputs)
        lstm_outputs = pad(lstm_outputs)
        lstm_outputs = self.lockeddrop(lstm_outputs)
        lstm_outputs = pack(lstm_outputs).data
        logits = pad(self.tag_clf(lstm_outputs)).contiguous()
        loss, trans = self.crit(logits, word_mask, tags)
        
        return loss, logits, trans

    def extract_static_embeddings(self, args, sents):
        processed = []
        if args.get('lowercase', True): # handle word case
            case = lambda x: x.lower()
        else:
            case = lambda x: x
        if args.get('char_lowercase', False): # handle character case
            char_case = lambda x: x.lower()
        else:
            char_case = lambda x: x
        for idx, sent in enumerate(sents):
            processed_sent = [self.vocab['word'].map([case(w[0]) for w in sent])]
            processed.append(processed_sent[0])
        
        words = get_long_tensor(processed, len(sents))
        words_mask = torch.eq(words, PAD_ID)

        return words, words_mask

