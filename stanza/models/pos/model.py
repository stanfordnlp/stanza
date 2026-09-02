import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence, PackedSequence

from stanza.models.common.bert_embedding import extract_bert_embeddings
from stanza.models.common.biaffine import BiaffineScorer
from stanza.models.common.foundation_cache import load_bert, load_charlm
from stanza.models.common.hlstm import HighwayLSTM
from stanza.models.common.dropout import WordDropout
from stanza.models.common.utils import attach_bert_model
from stanza.models.common.vocab import CompositeVocab
from stanza.models.common.char_model import CharacterModel
from stanza.models.common import utils
from stanza.models.pos.tag_columns import (TAG_LINK_EMB, TAG_LINK_HIDDEN, TAG_LINKS,
                                            tag_column_eval_order, tag_columns_from_args,
                                            validate_tag_columns)

logger = logging.getLogger('stanza')

# Older model files predate the tag_columns config entry and name the
# xpos and feats heads directly.  Map them onto the ModuleDict names so
# a released model still loads.
LEGACY_STATE_DICT_PREFIXES = (("xpos_hid.",    "tag_hid.xpos."),
                              ("xpos_clf.",    "tag_clf.xpos."),
                              ("ufeats_hid.",  "tag_hid.feats."),
                              ("ufeats_clf.",  "tag_clf.feats."))

def remap_legacy_state_dict(state_dict):
    """
    Rename the pre-ModuleDict head parameters, leaving anything else alone

    Returns the state dict unchanged if it has no legacy names, so this
    is safe to call on every load.
    """
    if not any(key.startswith(old) for key in state_dict for old, _ in LEGACY_STATE_DICT_PREFIXES):
        return state_dict

    logger.debug("Remapping legacy POS tagger head names in the saved model")
    remapped = type(state_dict)()
    for key, value in state_dict.items():
        for old, new in LEGACY_STATE_DICT_PREFIXES:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        remapped[key] = value
    return remapped

class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False, foundation_cache=None, bert_model=None, bert_tokenizer=None, force_bert_saved=False, peft_name=None):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        # UPOS is the root of the tag columns: every other column is
        # conditioned on it, or on something which is
        self.tag_columns = validate_tag_columns(tag_columns_from_args(args))
        self.tag_names = [x.name for x in self.tag_columns]
        self.tag_index = {name: idx for idx, name in enumerate(self.tag_names)}
        self.column_parents = {x.name: x.parents for x in self.tag_columns}
        # the order the heads are computed in, which is not the order
        # they are declared in when a column is conditioned on one
        # declared after it
        self.eval_order = [x for x in tag_column_eval_order(self.tag_columns) if x != 'upos']

        self.link_mode = args.get('tag_column_link', TAG_LINK_EMB)
        if self.link_mode not in TAG_LINKS:
            raise ValueError("Unknown tag column link '%s', expected one of %s" % (self.link_mode, TAG_LINKS))
        self.detach_parents = args.get('detach_parent_tags', False)

        if share_hid and any(x.parents not in ((), ('upos',)) for x in self.tag_columns):
            raise ValueError("share_hid has no per-column hidden layers and no tag embeddings, so it cannot condition a column on anything but upos")

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']

        if not share_hid:
            # upos embeddings
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                if args['charlm_forward_file'] is None or not os.path.exists(args['charlm_forward_file']):
                    raise FileNotFoundError('Could not find forward character model: {}  Please specify with --charlm_forward_file'.format(args['charlm_forward_file']))
                if args['charlm_backward_file'] is None or not os.path.exists(args['charlm_backward_file']):
                    raise FileNotFoundError('Could not find backward character model: {}  Please specify with --charlm_backward_file'.format(args['charlm_backward_file']))
                logger.debug("POS model loading charmodels: %s and %s", args['charlm_forward_file'], args['charlm_backward_file'])
                self.add_unsaved_module('charmodel_forward', load_charlm(args['charlm_forward_file'], foundation_cache=foundation_cache))
                self.add_unsaved_module('charmodel_backward', load_charlm(args['charlm_backward_file'], foundation_cache=foundation_cache))
                # optionally add a input transformation layer
                if self.args.get('charlm_transform_dim', 0):
                    self.charmodel_forward_transform = nn.Linear(self.charmodel_forward.hidden_dim(), self.args['charlm_transform_dim'], bias=False)
                    self.charmodel_backward_transform = nn.Linear(self.charmodel_backward.hidden_dim(), self.args['charlm_transform_dim'], bias=False)
                    input_size += self.args['charlm_transform_dim'] * 2
                else:
                    self.charmodel_forward_transform = None
                    self.charmodel_backward_transform = None
                    input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
            else:
                bidirectional = args.get('char_bidirectional', False)
                self.charmodel = CharacterModel(args, vocab, bidirectional=bidirectional)
                if bidirectional:
                    self.trans_char = nn.Linear(self.args['char_hidden_dim'] * 2, self.args['transformed_dim'], bias=False)
                else:
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
        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        if share_hid:
            clf_constructor = lambda insize, parent_size, outsize: nn.Linear(insize, outsize)
        else:
            clf_constructor = lambda insize, parent_size, outsize: BiaffineScorer(insize, parent_size, outsize)

        def hidden_size(name):
            """the width of a column's hidden layer, which is also what it offers a child in HIDDEN mode"""
            if name == 'upos':
                return self.args['deep_biaff_hidden_dim']
            if isinstance(vocab[name], CompositeVocab):
                return self.args['composite_deep_biaff_hidden_dim']
            return self.args['deep_biaff_hidden_dim']

        # in TAG_EMB mode a parent is represented by an embedding of its
        # tag, so anything used as a parent needs an embedding table.
        # upos already has one above
        self.tag_emb = nn.ModuleDict()
        if not share_hid and self.link_mode == TAG_LINK_EMB:
            for column in self.tag_columns:
                for parent in column.parents:
                    if parent == 'upos' or parent in self.tag_emb:
                        continue
                    if isinstance(vocab[parent], CompositeVocab):
                        raise ValueError("Tag column '%s' cannot be a parent with the %s link, as it has more than one tag per word.  Use the %s link instead" %
                                         (parent, TAG_LINK_EMB, TAG_LINK_HIDDEN))
                    self.tag_emb[parent] = nn.Embedding(len(vocab[parent]), self.args['tag_emb_dim'], padding_idx=0)

        def parent_size(name):
            if self.link_mode == TAG_LINK_EMB:
                return self.args['tag_emb_dim'] * len(self.column_parents[name])
            return sum(hidden_size(x) for x in self.column_parents[name])

        # every column after upos gets its own hidden layer and its own
        # classifier, keyed by name.  A ModuleDict rather than a list so
        # that the saved parameter names are tied to the column names -
        # tag_clf.xpos rather than tag_clf.0 - and a column's weights
        # can only ever be loaded back into that same column.  With a
        # list, two models whose columns are declared in a different
        # order would load each other's heads into the wrong place,
        # silently when the two tagsets happen to be the same size.
        self.tag_hid = nn.ModuleDict()
        self.tag_clf = nn.ModuleDict()
        for name in self.tag_names[1:]:
            composite = isinstance(vocab[name], CompositeVocab)
            if share_hid:
                # there is no hidden layer of its own: everything hangs
                # off upos_hid, so that is the input size
                insize = self.args['deep_biaff_hidden_dim']
            else:
                insize = self.args['composite_deep_biaff_hidden_dim'] if composite else self.args['deep_biaff_hidden_dim']
                self.tag_hid[name] = nn.Linear(self.args['hidden_dim'] * 2, insize)

            psize = parent_size(name)
            if composite:
                clf = nn.ModuleList([clf_constructor(insize, psize, l) for l in vocab[name].lens()])
                sub_clfs = clf
            else:
                clf = clf_constructor(insize, psize, len(vocab[name]))
                sub_clfs = [clf]
            if share_hid:
                for sub_clf in sub_clfs:
                    sub_clf.weight.data.zero_()
                    sub_clf.bias.data.zero_()
            self.tag_clf[name] = clf

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=0) # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def log_norms(self):
        utils.log_norms(self)

    def forward(self, word, word_mask, wordchars, wordchars_mask, tags, pretrained, word_orig_idx, sentlens, wordlens, text):
        """
        tags is one entry per tag column, in the order of self.tag_names

        An entry is None when the batch has no gold labels for that
        column, in which case that head still predicts but contributes
        nothing to the loss.  A short list (or None) is treated as all
        None, which is what prediction time looks like.
        """
        if tags is None:
            tags = []
        tags = list(tags) + [None] * (len(self.tag_names) - len(tags))

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, inputs[0].batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                all_forward_chars = self.charmodel_forward.build_char_representation(text)
                assert isinstance(all_forward_chars, list)
                if self.charmodel_forward_transform is not None:
                    all_forward_chars = [self.charmodel_forward_transform(x) for x in all_forward_chars]
                all_forward_chars = pack(pad_sequence(all_forward_chars, batch_first=True))

                all_backward_chars = self.charmodel_backward.build_char_representation(text)
                if self.charmodel_backward_transform is not None:
                    all_backward_chars = [self.charmodel_backward_transform(x) for x in all_backward_chars]
                all_backward_chars = pack(pad_sequence(all_backward_chars, batch_first=True))

                inputs += [all_forward_chars, all_backward_chars]
            else:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
                inputs += [char_reps]

        if self.bert_model is not None:
            device = next(self.parameters()).device
            processed_bert = extract_bert_embeddings(self.args['bert_model'], self.bert_tokenizer, self.bert_model, text, device, keep_endpoints=False,
                                                     detach=not self.args.get('bert_finetune', False) or not self.training,
                                                     bert_layer_mix=self.bert_layer_mix if self.bert_layer_mix is not None else None,
                                                     peft_name=self.peft_name)

            processed_bert = pad_sequence(processed_bert, batch_first=True)
            inputs += [pack(processed_bert)]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data

        upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        upos_pred = self.upos_clf(self.drop(upos_hid))

        upos = tags[0]
        if upos is not None:
            upos = pack(upos).data
            loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))
        else:
            loss = 0.0

        # what each column leaves behind for the ones conditioned on it:
        # its hidden layer, and, for a column with one tag per word, the
        # scores it predicted.  The scores are only ever read to pick the
        # most likely tag to embed, for a batch which has no gold tag to
        # embed instead; they are not themselves fed to a child
        hids = {'upos': upos_hid}
        tag_scores = {'upos': upos_pred}
        gold = {'upos': upos}
        predictions = {'upos': pad(upos_pred).max(2)[1]}

        for name in self.eval_order:
            tag = tags[self.tag_index[name]]
            if tag is not None:
                tag = pack(tag).data
            gold[name] = tag

            hid = upos_hid if self.share_hid else F.relu(self.tag_hid[name](self.drop(lstm_outputs)))
            hids[name] = hid

            if self.share_hid:
                clffunc = lambda clf, hid: clf(self.drop(hid))
            else:
                parent = self.parent_representation(name, hids, tag_scores, gold)
                clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(parent))

            if isinstance(self.vocab[name], CompositeVocab):
                tag_preds = []
                for i in range(len(self.vocab[name])):
                    tag_pred = clffunc(self.tag_clf[name][i], hid)
                    if tag is not None:
                        loss += self.crit(tag_pred.view(-1, tag_pred.size(-1)), tag[:, i].view(-1))
                    tag_preds.append(pad(tag_pred).max(2, keepdim=True)[1])
                predictions[name] = torch.cat(tag_preds, 2)
            else:
                tag_pred = clffunc(self.tag_clf[name], hid)
                tag_scores[name] = tag_pred
                if tag is not None:
                    loss += self.crit(tag_pred.view(-1, tag_pred.size(-1)), tag.view(-1))
                predictions[name] = pad(tag_pred).max(2)[1]

        # back into the declared order, which is the order the tags came
        # in and the order the caller unmaps them in
        preds = [predictions[name] for name in self.tag_names]

        return loss, preds

    def parent_representation(self, name, hids, tag_scores, gold):
        """
        Build what a column is conditioned on, from the columns it hangs off

        With the hidden link, a parent hands over its hidden layer, and
        the gradient from this column travels back into that parent's
        head unless detach_parent_tags is set.  With the tag embedding
        link, a parent hands over an embedding of its tag - the gold one
        while training, when this batch has it, and otherwise the one it
        just predicted.
        """
        reps = []
        for parent in self.column_parents[name]:
            if self.link_mode == TAG_LINK_HIDDEN:
                rep = hids[parent]
                if self.detach_parents:
                    rep = rep.detach()
            else:
                tag = gold.get(parent)
                if self.training and tag is not None:
                    tag_ids = tag
                else:
                    tag_ids = tag_scores[parent].max(1)[1]
                rep = self.upos_emb(tag_ids) if parent == 'upos' else self.tag_emb[parent](tag_ids)
            reps.append(rep)
        return reps[0] if len(reps) == 1 else torch.cat(reps, 1)
