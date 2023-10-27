import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence, PackedSequence

from stanza.models.common.data import map_to_ids, get_long_tensor
from stanza.models.common.packed_lstm import PackedLSTM
from stanza.models.common.dropout import WordDropout, LockedDropout
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from stanza.models.common.crf import CRFLoss
from stanza.models.common.foundation_cache import load_bert
from stanza.models.common.vocab import PAD_ID, UNK_ID, EMPTY_ID
from stanza.models.common.bert_embedding import extract_bert_embeddings

logger = logging.getLogger('stanza')

class NERTagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, foundation_cache=None, force_bert_saved=False):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            emb_finetune = self.args.get('emb_finetune', True)

            # load pretrained embeddings if specified
            word_emb = nn.Embedding(len(self.vocab['word']), self.args['word_emb_dim'], PAD_ID)
            # if a model trained with no 'delta' vocab is loaded, and
            # emb_finetune is off, any resaving of the model will need
            # the updated vectors.  this is accounted for in load()
            if not emb_finetune or 'delta' in self.vocab:
                # if emb_finetune is off
                # or if the delta embedding is present
                # then we won't fine tune the original embedding
                add_unsaved_module('word_emb', word_emb)
                self.word_emb.weight.detach_()
            else:
                self.word_emb = word_emb
            if emb_matrix is not None:
                self.init_emb(emb_matrix)

            # TODO: allow for expansion of delta embedding if new
            # training data has new words in it?
            self.delta_emb = None
            if 'delta' in self.vocab:
                # zero inits seems to work better
                # note that the gradient will flow to the bottom and then adjust the 0 weights
                # as opposed to a 0 matrix cutting off the gradient if higher up in the model
                self.delta_emb = nn.Embedding(len(self.vocab['delta']), self.args['word_emb_dim'], PAD_ID)
                nn.init.zeros_(self.delta_emb.weight)
                # if the model was trained with a delta embedding, but emb_finetune is off now,
                # then we will detach the delta embedding
                if not emb_finetune:
                    self.delta_emb.weight.detach_()

            input_size += self.args['word_emb_dim']

        # TODO: this, pos, depparse should all be refactored
        # FIXME: possibly pos and depparse are all losing a finetuned transformer if loaded & saved
        # (the force_bert_saved option here handles that)
        if self.args.get('bert_model', None):
            # first we load the transformer model and possibly turn off its requires_grad parameters ...
            if self.args.get('bert_finetune', False):
                bert_model, bert_tokenizer = load_bert(self.args['bert_model'])
            else:
                bert_model, bert_tokenizer = load_bert(self.args['bert_model'], foundation_cache)
                for n, p in bert_model.named_parameters():
                    p.requires_grad = False
            # then we attach it to the NER model
            # if force_bert_saved is True, that probably indicates the save file had a transformer in it
            # thus we need to save it again in the future to avoid losing it when resaving
            if self.args.get('bert_finetune', False) or force_bert_saved:
                self.bert_model = bert_model
                add_unsaved_module('bert_tokenizer', bert_tokenizer)
            else:
                add_unsaved_module('bert_model', bert_model)
                add_unsaved_module('bert_tokenizer', bert_tokenizer)
            input_size += self.bert_model.config.hidden_size
        else:
            self.bert_model = None
            self.bert_tokenizer = None

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
        tag_lengths = self.vocab['tag'].lens()
        self.num_output_layers = len(tag_lengths)
        if self.args.get('connect_output_layers'):
            tag_clfs = [nn.Linear(self.args['hidden_dim']*2, tag_lengths[0])]
            for prev_length, next_length in zip(tag_lengths[:-1], tag_lengths[1:]):
                tag_clfs.append(nn.Linear(self.args['hidden_dim']*2 + prev_length, next_length))
            self.tag_clfs = nn.ModuleList(tag_clfs)
        else:
            self.tag_clfs = nn.ModuleList([nn.Linear(self.args['hidden_dim']*2, num_tag) for num_tag in tag_lengths])
        for tag_clf in self.tag_clfs:
            tag_clf.bias.data.zero_()
        self.crits = nn.ModuleList([CRFLoss(num_tag) for num_tag in tag_lengths])

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

    def log_norms(self):
        lines = ["NORMS FOR MODEL PARAMTERS"]
        for name, param in self.named_parameters():
            if param.requires_grad and name.split(".")[0] not in ('charmodel_forward', 'charmodel_backward'):
                lines.append("  %s %.6g" % (name, torch.norm(param).item()))
        logger.info("\n".join(lines))

    def forward(self, sentences, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx):
        device = next(self.parameters()).device

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        
        inputs = []
        batch_size = len(sentences)

        if self.args['word_emb_dim'] > 0:
            #extract static embeddings
            static_words, word_mask = self.extract_static_embeddings(self.args, sentences, self.vocab['word'])

            word_mask = word_mask.to(device)
            static_words = static_words.to(device)
                
            word_static_emb = self.word_emb(static_words)

            if 'delta' in self.vocab and self.delta_emb is not None:
                # masks should be the same
                delta_words, _ = self.extract_static_embeddings(self.args, sentences, self.vocab['delta'])
                delta_words = delta_words.to(device)
                # unclear whether to treat words in the main embedding
                # but not in delta as unknown
                # simple heuristic though - treating them as not
                # unknown keeps existing models the same when
                # separating models into the base WV and delta WV
                # also, note that at training time, words like this
                # did not show up in the training data, but are
                # not exactly UNK, so it makes sense
                delta_unk_mask = torch.eq(delta_words, UNK_ID)
                static_unk_mask = torch.not_equal(static_words, UNK_ID)
                unk_mask = delta_unk_mask * static_unk_mask
                delta_words[unk_mask] = PAD_ID

                delta_emb = self.delta_emb(delta_words)
                word_static_emb = word_static_emb + delta_emb

            word_emb = pack(word_static_emb)
            inputs += [word_emb]

        if self.bert_model is not None:
            device = next(self.parameters()).device
            processed_bert = extract_bert_embeddings(self.args['bert_model'], self.bert_tokenizer, self.bert_model, sentences, device, keep_endpoints=False,
                                                     detach=not self.args.get('bert_finetune', False))
            processed_bert = pad_sequence(processed_bert, batch_first=True)
            inputs += [pack(processed_bert)]

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

        loss = 0
        logits = []
        trans = []
        for idx, (tag_clf, crit) in enumerate(zip(self.tag_clfs, self.crits)):
            if not self.args.get('connect_output_layers') or idx == 0:
                next_logits = pad(tag_clf(lstm_outputs)).contiguous()
            else:
                # here we pack the output of the previous round, then append it
                packed_logits = pack(next_logits).data
                input_logits = torch.cat([lstm_outputs, packed_logits], axis=1)
                next_logits = pad(tag_clf(input_logits)).contiguous()
            # the tag_mask lets us avoid backprop on a blank tag
            tag_mask = torch.eq(tags[:, :, idx], EMPTY_ID)
            next_loss, next_trans = crit(next_logits, torch.bitwise_or(tag_mask, word_mask), tags[:, :, idx])
            loss = loss + next_loss
            logits.append(next_logits)
            trans.append(next_trans)

        return loss, logits, trans

    @staticmethod
    def extract_static_embeddings(args, sents, vocab):
        processed = []
        if args.get('lowercase', True): # handle word case
            case = lambda x: x.lower()
        else:
            case = lambda x: x
        for idx, sent in enumerate(sents):
            processed_sent = [vocab.map([case(w) for w in sent])]
            processed.append(processed_sent[0])

        words = get_long_tensor(processed, len(sents))
        words_mask = torch.eq(words, PAD_ID)

        return words, words_mask

