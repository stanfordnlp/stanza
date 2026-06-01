import logging
import os

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence, PackedSequence

from stanza.models.common.bert_embedding import extract_bert_embeddings
from stanza.models.common.biaffine import DeepBiaffineScorer
from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from stanza.models.common.foundation_cache import load_charlm
from stanza.models.common.hlstm import HighwayLSTM
from stanza.models.common.dropout import WordDropout
from stanza.models.common.utils import attach_bert_model
from stanza.models.common.vocab import CompositeVocab, VOCAB_PREFIX_SIZE
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from stanza.models.common import utils

logger = logging.getLogger('stanza')

class BaseParser(nn.Module, ABC):
    def __init__(self, args, vocab):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []

    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def get_params(self, skip_modules):
        model_state = self.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.unsaved_modules]
            for k in skipped:
                del model_state[k]
        return model_state

    def load_params(self, checkpoint):
        return self.load_state_dict(checkpoint, strict=False)

    def log_norms(self):
        utils.log_norms(self)

    @abstractmethod
    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        """ Return loss & predictions for this batch of sentences """

    @abstractmethod
    def loss(self, *args, **kwargs):
        """ Return just the loss for this batch.  Should be a torch tensor """

    @abstractmethod
    def predict(self, *args, **kwargs):
        """ Return a list of predictions for each sentence, where each prediction is a list of (head, deprel) """

    def get_device(self):
        return next(self.parameters()).device

    def empty_stats(self):
        """
        by default, don't track any stats per batch.  the transition parser has some more interesting stats
        """
        return None

class EmbeddingParser(BaseParser):
    def __init__(self, args, vocab, emb_matrix=None, foundation_cache=None, bert_model=None, bert_tokenizer=None, force_bert_saved=False, peft_name=None):
        super().__init__(args, vocab)

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
                if self.args['charlm_forward_file'] is None or not os.path.exists(self.args['charlm_forward_file']):
                    raise FileNotFoundError('Could not find forward character model: {}  Please specify with --charlm_forward_file'.format(self.args['charlm_forward_file']))
                if self.args['charlm_backward_file'] is None or not os.path.exists(self.args['charlm_backward_file']):
                    raise FileNotFoundError('Could not find backward character model: {}  Please specify with --charlm_backward_file'.format(self.args['charlm_backward_file']))
                logger.debug("Depparse model loading charmodels: %s and %s", self.args['charlm_forward_file'], self.args['charlm_backward_file'])
                self.add_unsaved_module('charmodel_forward', load_charlm(self.args['charlm_forward_file'], foundation_cache=foundation_cache))
                self.add_unsaved_module('charmodel_backward', load_charlm(self.args['charlm_backward_file'], foundation_cache=foundation_cache))
                input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
            else:
                self.charmodel = CharacterModel(self.args, vocab)
                self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
                input_size += self.args['transformed_dim']

        self.peft_name = peft_name
        attach_bert_model(self, bert_model, bert_tokenizer, self.args.get('use_peft', False), force_bert_saved)
        if self.args.get('bert_model', None):
            # TODO: refactor bert_hidden_layers between the different models
            if self.args.get('bert_hidden_layers', False):
                # The average will be offset by 1/N so that the default zeros
                # represents an average of the N layers
                self.bert_layer_mix = nn.Linear(self.args['bert_hidden_layers'], 1, bias=False)
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

        self.input_size = input_size

        # recurrent layers
        self.parserlstm = HighwayLSTM(self.input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(self.input_size) / np.sqrt(self.input_size))
        self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        self.drop = nn.Dropout(self.args['dropout'])
        self.worddrop = WordDropout(self.args['word_dropout'])

        self.num_relations = len(vocab['deprel']) - VOCAB_PREFIX_SIZE

    def embed(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
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
            bert_finetuning = getattr(self, 'bert_finetuning', False)
            processed_bert = extract_bert_embeddings(self.args['bert_model'], self.bert_tokenizer, self.bert_model, text, device, keep_endpoints=True,
                                                     detach=not bert_finetuning or not self.training,
                                                     bert_layer_mix=self.bert_layer_mix if self.bert_layer_mix is not None else None,
                                                     peft_name=self.peft_name)

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

        return lstm_outputs

class GraphParser(EmbeddingParser):
    def __init__(self, args, vocab, emb_matrix=None, foundation_cache=None, bert_model=None, bert_tokenizer=None, force_bert_saved=False, peft_name=None):
        super().__init__(args, vocab, emb_matrix=emb_matrix, foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=force_bert_saved, peft_name=peft_name)

        # classifiers
        # args.get to preserve old models, including models other people might have created
        if self.args.get('use_arc_embedding'):
            logger.debug("Using arc embedding enhancement")
            self.arc_embedding = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], self.args['deep_biaff_output_dim'], pairwise=True, dropout=self.args['dropout'])
            self.unlabeled_linear = nn.Sequential(self.drop,
                                                  nn.Linear(self.args['deep_biaff_output_dim'], 1))
            self.deprel_linear = nn.Sequential(self.drop,
                                               nn.Linear(self.args['deep_biaff_output_dim'], 2 * self.args['deep_biaff_output_dim']),
                                               nn.ReLU(),
                                               self.drop,
                                               nn.Linear(self.args['deep_biaff_output_dim'] * 2, self.num_relations))
        else:
            logger.debug("Not using arc embedding enhancement")
            self.unlabeled = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=self.args['dropout'])
            self.deprel = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], self.num_relations, pairwise=True, dropout=self.args['dropout'])
        if self.args['linearization']:
            self.linearization = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=self.args['dropout'])
        if self.args['distance']:
            self.distance = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=self.args['dropout'])

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum') # ignore padding

    @staticmethod
    def decode_graph_predictions(vocab, batch_size, preds, sentlens):
        head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], sentlens)] # remove attachment for the root
        deprel_seqs = [vocab.unmap([preds[1][i][j+1][h] for j, h in enumerate(hs)]) for i, hs in enumerate(head_seqs)]
        pred_tokens = [[[head_seqs[i][j], deprel_seqs[i][j]] for j in range(sentlens[i]-1)] for i in range(batch_size)]
        return pred_tokens

    def forward_scores(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        lstm_outputs = self.embed(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)

        if self.args.get('use_arc_embedding'):
            arc_scores = self.arc_embedding(self.drop(lstm_outputs), self.drop(lstm_outputs))
            unlabeled_scores = self.unlabeled_linear(arc_scores).squeeze(3)
            deprel_scores = self.deprel_linear(arc_scores).squeeze(3)
        else:
            unlabeled_scores = self.unlabeled(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            deprel_scores = self.deprel(self.drop(lstm_outputs), self.drop(lstm_outputs))

        #goldmask = head.new_zeros(*head.size(), head.size(-1)+1, dtype=torch.uint8)
        #goldmask.scatter_(2, head.unsqueeze(2), 1)

        head_offset = None
        if self.args['linearization'] or self.args['distance']:
            head_offset = torch.arange(word.size(1), device=head.device).view(1, 1, -1).expand(word.size(0), -1, -1) - torch.arange(word.size(1), device=head.device).view(1, -1, 1).expand(word.size(0), -1, -1)

        lin_scores = None
        if self.args['linearization']:
            lin_scores = self.linearization(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            unlabeled_scores += F.logsigmoid(lin_scores * torch.sign(head_offset).float()).detach()

        dist_kld = None
        if self.args['distance']:
            dist_scores = self.distance(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            dist_pred = 1 + F.softplus(dist_scores)
            dist_target = torch.abs(head_offset)
            dist_kld = -torch.log((dist_target.float() - dist_pred)**2/2 + 1)
            unlabeled_scores += dist_kld.detach()

        diag = torch.eye(head.size(-1)+1, dtype=torch.bool, device=head.device).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))

        return unlabeled_scores, deprel_scores, head_offset, lin_scores, dist_kld

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        unlabeled_scores, deprel_scores, head_offset, lin_scores, dist_kld = self.forward_scores(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)

        preds = []
        if self.training:
            unlabeled_scores = unlabeled_scores[:, 1:, :] # exclude attachment for the root symbol
            unlabeled_scores = unlabeled_scores.masked_fill(word_mask.unsqueeze(1), -float('inf'))
            unlabeled_target = head.masked_fill(word_mask[:, 1:], -1)
            loss = self.crit(unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)), unlabeled_target.view(-1))

            deprel_scores = deprel_scores[:, 1:] # exclude attachment for the root symbol
            #deprel_scores = deprel_scores.masked_select(goldmask.unsqueeze(3)).view(-1, len(self.vocab['deprel']))
            deprel_scores = torch.gather(deprel_scores, 2, head.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, self.num_relations)).view(-1, self.num_relations)
            deprel_target = deprel - VOCAB_PREFIX_SIZE
            deprel_target = deprel_target.masked_fill(word_mask[:, 1:], -1)
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
                # dist_kld[:, 1:] so that the root isn't included in the distance calculation
                dist_kld = torch.gather(dist_kld[:, 1:], 2, head.unsqueeze(2))
                loss -= dist_kld.sum()

            loss /= wordchars.size(0) # number of words
        else:
            loss = 0
            preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
            preds.append(deprel_scores.max(3)[1].detach().cpu().numpy() + VOCAB_PREFIX_SIZE)

        return loss, preds

    def loss(self, *args, **kwargs):
        """
        returns the loss of applying forward() to this batch

        currently returns no stats - could return something other than None to track batch stats
        """
        loss, _ = self.forward(*args, **kwargs)
        return loss, None

    def predict(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        batch_size = word.size(0)
        _, preds = self(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        pred_tokens = self.decode_graph_predictions(self.vocab['deprel'], batch_size, preds, sentlens)
        return pred_tokens

class EnsembleGraphParser(BaseParser):
    def __init__(self, args, vocab, models):
        super().__init__(args, vocab)
        self.models = nn.ModuleList(models)

    def get_params(self, skip_modules):
        params = []
        args = []
        for model in self.models:
            params.append(model.get_params(skip_modules))
            config = dict(model.args)
            # sanitize enums for torch.load(weights_only=True)
            if 'transition_subtree_combination' in config:
                config['transition_subtree_combination'] = config['transition_subtree_combination'].name
            args.append(config)
        checkpoint = {
            "num_models": len(self.models),
            "params": params,
            "args": args,
        }
        return checkpoint

    def load_params(self, checkpoint):
        for model, params in zip(self.models, checkpoint["params"]):
            model.load_params(params)

    def loss(self, *args, **kwargs):
        raise NotImplementedError("Cannot train ensemble parser")

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        unlabeled_scores = 0
        deprel_scores = 0
        for model in self.models:
            model_unlabeled_scores, model_deprel_scores, _, _, _ = model.forward_scores(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
            unlabeled_scores += model_unlabeled_scores
            deprel_scores += model_deprel_scores

        if self.training:
            raise NotImplementedError("Cannot train ensemble parser")
        else:
            loss = 0
            preds = []
            preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
            preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())

        return loss, preds

    # TODO: refactor this?
    def predict(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        batch_size = word.size(0)
        _, preds = self(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        pred_tokens = GraphParser.decode_graph_predictions(self.vocab['deprel'], batch_size, preds, sentlens)
        return pred_tokens

    def get_device(self):
        return self.models[0].get_device()

