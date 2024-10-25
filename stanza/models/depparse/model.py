import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence, PackedSequence

from stanza.models.common.bert_embedding import extract_bert_embeddings
from stanza.models.common.biaffine import DeepBiaffineScorer
from stanza.models.common.foundation_cache import load_charlm
from stanza.models.common.hlstm import HighwayLSTM
from stanza.models.common.dropout import WordDropout
from stanza.models.common.utils import attach_bert_model
from stanza.models.common.vocab import CompositeVocab
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from stanza.models.common import utils

logger = logging.getLogger('stanza')

class Parser(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False, foundation_cache=None, bert_model=None, bert_tokenizer=None, force_bert_saved=False, peft_name=None):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            self.lemma_emb = nn.Embedding(len(vocab['lemma']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim'] * 2

        if self.args['tag_emb_dim'] > 0:
            if self.args.get('use_upos', True):
                self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)
            if self.args.get('use_xpos', True):
                if not isinstance(vocab['xpos'], CompositeVocab):
                    self.xpos_emb = nn.Embedding(len(vocab['xpos']), self.args['tag_emb_dim'], padding_idx=0)
                else:
                    self.xpos_emb = nn.ModuleList()

                    for l in vocab['xpos'].lens():
                        self.xpos_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))
            if self.args.get('use_upos', True) or self.args.get('use_xpos', True):
                input_size += self.args['tag_emb_dim']

            if self.args.get('use_ufeats', True):
                self.ufeats_emb = nn.ModuleList()

                for l in vocab['feats'].lens():
                    self.ufeats_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))

                input_size += self.args['tag_emb_dim']

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                if args['charlm_forward_file'] is None or not os.path.exists(args['charlm_forward_file']):
                    raise FileNotFoundError('Could not find forward character model: {}  Please specify with --charlm_forward_file'.format(args['charlm_forward_file']))
                if args['charlm_backward_file'] is None or not os.path.exists(args['charlm_backward_file']):
                    raise FileNotFoundError('Could not find backward character model: {}  Please specify with --charlm_backward_file'.format(args['charlm_backward_file']))
                logger.debug("Depparse model loading charmodels: %s and %s", args['charlm_forward_file'], args['charlm_backward_file'])
                self.add_unsaved_module('charmodel_forward', load_charlm(args['charlm_forward_file'], foundation_cache=foundation_cache))
                self.add_unsaved_module('charmodel_backward', load_charlm(args['charlm_backward_file'], foundation_cache=foundation_cache))
                input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
            else:
                self.charmodel = CharacterModel(args, vocab)
                self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
                input_size += self.args['transformed_dim']

        self.peft_name = peft_name
        attach_bert_model(self, bert_model, bert_tokenizer, self.args.get('use_peft', False), force_bert_saved)
        if self.args.get('bert_model', None):
            # TODO: refactor bert_hidden_layers between the different models
            if args.get('bert_hidden_layers', False):
                # The average will be offset by 1/N so that the default zeros
                # represents an average of the N layers
                self.bert_layer_mix = nn.Linear(args['bert_hidden_layers'], 1, bias=False)
                nn.init.zeros_(self.bert_layer_mix.weight)
            else:
                # an average of layers 2, 3, 4 will be used
                # (for historic reasons)
                self.bert_layer_mix = None
            input_size += self.bert_model.config.hidden_size

        if self.args['pretrain']:
            # pretrained embeddings, by default this won't be saved into model file
            self.add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(emb_matrix, freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        # recurrent layers
        self.parserlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers
        self.unlabeled = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])
        self.deprel = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], len(vocab['deprel']), pairwise=True, dropout=args['dropout'])
        if args['linearization']:
            self.linearization = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])
        if args['distance']:
            self.distance = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum') # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def log_norms(self):
        utils.log_norms(self)

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        #def pad(x):
        #    return pad_packed_sequence(PackedSequence(x, pretrained_emb.batch_sizes), batch_first=True)[0]

        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            lemma_emb = self.lemma_emb(lemma)
            lemma_emb = pack(lemma_emb)
            inputs += [word_emb, lemma_emb]

        if self.args['tag_emb_dim'] > 0:
            if self.args.get('use_upos', True):
                pos_emb = self.upos_emb(upos)
            else:
                pos_emb = 0

            if self.args.get('use_xpos', True):
                if isinstance(self.vocab['xpos'], CompositeVocab):
                    for i in range(len(self.vocab['xpos'])):
                        pos_emb += self.xpos_emb[i](xpos[:, :, i])
                else:
                    pos_emb += self.xpos_emb(xpos)

            if self.args.get('use_upos', True) or self.args.get('use_xpos', True):
                pos_emb = pack(pos_emb)
                inputs += [pos_emb]

            if self.args.get('use_ufeats', True):
                feats_emb = 0
                for i in range(len(self.vocab['feats'])):
                    feats_emb += self.ufeats_emb[i](ufeats[:, :, i])
                feats_emb = pack(feats_emb)

                inputs += [pos_emb]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                # \n is to add a somewhat neutral "word" for the ROOT
                charlm_text = [["\n"] + x for x in text]
                all_forward_chars = self.charmodel_forward.build_char_representation(charlm_text)
                all_forward_chars = pack(pad_sequence(all_forward_chars, batch_first=True))
                all_backward_chars = self.charmodel_backward.build_char_representation(charlm_text)
                all_backward_chars = pack(pad_sequence(all_backward_chars, batch_first=True))
                inputs += [all_forward_chars, all_backward_chars]
            else:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
                inputs += [char_reps]

        if self.bert_model is not None:
            device = next(self.parameters()).device
            processed_bert = extract_bert_embeddings(self.args['bert_model'], self.bert_tokenizer, self.bert_model, text, device, keep_endpoints=True,
                                                     num_layers=self.bert_layer_mix.in_features if self.bert_layer_mix is not None else None,
                                                     detach=not self.args.get('bert_finetune', False) or not self.training,
                                                     peft_name=self.peft_name)
            if self.bert_layer_mix is not None:
                # use a linear layer to weighted average the embedding dynamically
                processed_bert = [self.bert_layer_mix(feature).squeeze(2) + feature.sum(axis=2) / self.bert_layer_mix.in_features for feature in processed_bert]

            # we are using the first endpoint from the transformer as the "word" for ROOT
            processed_bert = [x[:-1, :] for x in processed_bert]
            processed_bert = pad_sequence(processed_bert, batch_first=True)
            inputs += [pack(processed_bert)]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)

        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)

        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = self.parserlstm(lstm_inputs, sentlens, hx=(self.parserlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.parserlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)

        unlabeled_scores = self.unlabeled(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
        deprel_scores = self.deprel(self.drop(lstm_outputs), self.drop(lstm_outputs))

        #goldmask = head.new_zeros(*head.size(), head.size(-1)+1, dtype=torch.uint8)
        #goldmask.scatter_(2, head.unsqueeze(2), 1)

        if self.args['linearization'] or self.args['distance']:
            head_offset = torch.arange(word.size(1), device=head.device).view(1, 1, -1).expand(word.size(0), -1, -1) - torch.arange(word.size(1), device=head.device).view(1, -1, 1).expand(word.size(0), -1, -1)

        if self.args['linearization']:
            lin_scores = self.linearization(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            unlabeled_scores += F.logsigmoid(lin_scores * torch.sign(head_offset).float()).detach()

        if self.args['distance']:
            dist_scores = self.distance(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            dist_pred = 1 + F.softplus(dist_scores)
            dist_target = torch.abs(head_offset)
            dist_kld = -torch.log((dist_target.float() - dist_pred)**2/2 + 1)
            unlabeled_scores += dist_kld.detach()

        diag = torch.eye(head.size(-1)+1, dtype=torch.bool, device=head.device).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))

        preds = []

        if self.training:
            unlabeled_scores = unlabeled_scores[:, 1:, :] # exclude attachment for the root symbol
            unlabeled_scores = unlabeled_scores.masked_fill(word_mask.unsqueeze(1), -float('inf'))
            unlabeled_target = head.masked_fill(word_mask[:, 1:], -1)
            loss = self.crit(unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)), unlabeled_target.view(-1))

            deprel_scores = deprel_scores[:, 1:] # exclude attachment for the root symbol
            #deprel_scores = deprel_scores.masked_select(goldmask.unsqueeze(3)).view(-1, len(self.vocab['deprel']))
            deprel_scores = torch.gather(deprel_scores, 2, head.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, len(self.vocab['deprel']))).view(-1, len(self.vocab['deprel']))
            deprel_target = deprel.masked_fill(word_mask[:, 1:], -1)
            loss += self.crit(deprel_scores.contiguous(), deprel_target.view(-1))

            if self.args['linearization']:
                #lin_scores = lin_scores[:, 1:].masked_select(goldmask)
                lin_scores = torch.gather(lin_scores[:, 1:], 2, head.unsqueeze(2)).view(-1)
                lin_scores = torch.cat([-lin_scores.unsqueeze(1)/2, lin_scores.unsqueeze(1)/2], 1)
                #lin_target = (head_offset[:, 1:] > 0).long().masked_select(goldmask)
                lin_target = torch.gather((head_offset[:, 1:] > 0).long(), 2, head.unsqueeze(2))
                loss += self.crit(lin_scores.contiguous(), lin_target.view(-1))

            if self.args['distance']:
                #dist_kld = dist_kld[:, 1:].masked_select(goldmask)
                dist_kld = torch.gather(dist_kld[:, 1:], 2, head.unsqueeze(2))
                loss -= dist_kld.sum()

            loss /= wordchars.size(0) # number of words
        else:
            loss = 0
            preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
            preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())

        return loss, preds
