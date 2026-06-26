"""
A trainer class to handle training and testing of models.
"""

import gzip
import io
import os
import sys
import numpy as np
from collections import Counter, defaultdict
import logging
import pickle
import torch
from torch import nn
import torch.nn.init as init

import stanza.models.common.seq2seq_constant as constant
from stanza.models.common.doc import TEXT, UPOS
from stanza.models.common.foundation_cache import load_charlm
from stanza.models.common.seq2seq_model import Seq2SeqModel
from stanza.models.common.char_model import CharacterLanguageModelWordAdapter
from stanza.models.common import utils, loss
from stanza.models.lemma import edit
from stanza.models.lemma.vocab import MultiVocab
from stanza.models.lemma_classifier.base_model import LemmaClassifier

logger = logging.getLogger('stanza')

# Sentinel key in pos_dict for the pos-independent word_dict fallback
_POS_INDEPENDENT = "*"

# Version tag stored in the checkpoint to distinguish dict formats
_DICTS_VERSION_LEGACY = 1   # old (word_dict, composite_dict) tuple
_DICTS_VERSION_POS    = 2   # new {pos: {word: lemma}}, gzip-pickled bytes


def unpack_batch(batch, device):
    """ Unpack a batch from the data loader. """
    inputs = [b.to(device) if b is not None else None for b in batch[:6]]
    orig_idx = batch[6]
    text = batch[7]
    return inputs, orig_idx, text


def _pack_pos_dict(pos_dict):
    """Serialize pos_dict to gzip-compressed pickle bytes for storage."""
    raw = pickle.dumps(pos_dict, protocol=4)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as gz:
        gz.write(raw)
    return buf.getvalue()


def _unpack_pos_dict(data):
    """Deserialize pos_dict from gzip-compressed pickle bytes."""
    buf = io.BytesIO(data)
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        raw = gz.read()
    return pickle.loads(raw)


def _legacy_dicts_to_pos_dict(word_dict, composite_dict):
    """
    Convert the old (word_dict, composite_dict) pair to the new pos_dict format.

    word_dict entries become pos_dict[_POS_INDEPENDENT][word] = lemma.
    composite_dict entries that *differ* from word_dict are stored under their
    POS tag; redundant entries (99% for Slovenian) are dropped.
    """
    pos_dict = defaultdict(dict)
    for w, l in word_dict.items():
        pos_dict[_POS_INDEPENDENT][w] = l
    for (w, pos), l in composite_dict.items():
        if word_dict.get(w) != l:
            pos_dict[pos][w] = l
    return dict(pos_dict)


class Trainer(object):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, emb_matrix=None, model_file=None, device=None, foundation_cache=None, lemma_classifier_args=None):
        if model_file is not None:
            # load everything from file
            self.load(model_file, args, foundation_cache, lemma_classifier_args)
        else:
            # build model from scratch
            self.args = args
            if args['dict_only']:
                self.model = None
            else:
                self.model = self.build_seq2seq(args, emb_matrix, foundation_cache)
            self.vocab = vocab
            # dict-based component: {pos: {word: lemma}}
            # _POS_INDEPENDENT ("*") holds the pos-agnostic fallback
            self.pos_dict = {}
            self.contextual_lemmatizers = []

        self.caseless = self.args.get('caseless', False)

        if not self.args['dict_only']:
            self.model = self.model.to(device)
            if self.args.get('edit', False):
                self.crit = loss.MixLoss(self.vocab['char'].size, self.args['alpha']).to(device)
                logger.debug("Running seq2seq lemmatizer with edit classifier...")
            else:
                self.crit = loss.SequenceLoss(self.vocab['char'].size).to(device)
            self.optimizer = utils.get_optimizer(self.args['optim'], self.model, self.args['lr'])

    def build_seq2seq(self, args, emb_matrix, foundation_cache):
        charmodel = None
        charlms = []
        if args is not None and args.get('charlm') and args.get('charlm_forward_file', None):
            charmodel_forward = load_charlm(args['charlm_forward_file'], foundation_cache=foundation_cache)
            charlms.append(charmodel_forward)
        if args is not None and args.get('charlm') and args.get('charlm_backward_file', None):
            charmodel_backward = load_charlm(args['charlm_backward_file'], foundation_cache=foundation_cache)
            charlms.append(charmodel_backward)
        if len(charlms) > 0:
            charlms = nn.ModuleList(charlms)
            charmodel = CharacterLanguageModelWordAdapter(charlms)
        model = Seq2SeqModel(args, emb_matrix=emb_matrix, contextual_embedding=charmodel)
        return model

    def update(self, batch, eval=False):
        device = next(self.model.parameters()).device
        inputs, orig_idx, text = unpack_batch(batch, device)
        src, src_mask, tgt_in, tgt_out, pos, edits = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs, edit_logits = self.model(src, src_mask, tgt_in, pos, raw=text)
        if self.args.get('edit', False):
            assert edit_logits is not None
            loss = self.crit(log_probs.view(-1, self.vocab['char'].size), tgt_out.view(-1), \
                    edit_logits, edits)
        else:
            loss = self.crit(log_probs.view(-1, self.vocab['char'].size), tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, beam_size=1, vocab=None):
        if vocab is None:
            vocab = self.vocab

        device = next(self.model.parameters()).device
        inputs, orig_idx, text = unpack_batch(batch, device)
        src, src_mask, tgt, tgt_mask, pos, edits = inputs

        self.model.eval()
        batch_size = src.size(0)
        preds, edit_logits = self.model.predict(src, src_mask, pos=pos, beam_size=beam_size, raw=text)
        pred_seqs = [vocab['char'].unmap(ids) for ids in preds] # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs] # join chars to be tokens
        pred_tokens = utils.unsort(pred_tokens, orig_idx)
        if self.args.get('edit', False):
            assert edit_logits is not None
            edits = np.argmax(edit_logits.data.cpu().numpy(), axis=1).reshape([batch_size]).tolist()
            edits = utils.unsort(edits, orig_idx)
        else:
            edits = None
        return pred_tokens, edits

    def postprocess(self, words, preds, edits=None):
        """ Postprocess, mainly for handing edits. """
        assert len(words) == len(preds), "Lemma predictions must have same length as words."
        edited = []
        if self.args.get('edit', False):
            assert edits is not None and len(words) == len(edits)
            for w, p, e in zip(words, preds, edits):
                lem = edit.edit_word(w, p, e)
                edited += [lem]
        else:
            edited = preds # do not edit
        # final sanity check
        assert len(edited) == len(words)
        final = []
        for lem, w in zip(edited, words):
            if len(lem) == 0 or constant.UNK in lem:
                final += [w] # invalid prediction, fall back to word
            else:
                final += [lem]
        return final

    def has_contextual_lemmatizers(self):
        return self.contextual_lemmatizers is not None and len(self.contextual_lemmatizers) > 0

    def predict_contextual(self, sentence_words, sentence_tags, preds):
        if len(self.contextual_lemmatizers) == 0:
            return preds

        # reversed so that the first lemmatizer has priority
        for contextual in reversed(self.contextual_lemmatizers):
            pred_idx = []
            pred_sent_words = []
            pred_sent_tags = []
            pred_sent_ids = []
            for sent_id, (words, tags) in enumerate(zip(sentence_words, sentence_tags)):
                indices = contextual.target_indices(words, tags)
                for idx in indices:
                    pred_idx.append(idx)
                    pred_sent_words.append(words)
                    pred_sent_tags.append(tags)
                    pred_sent_ids.append(sent_id)
            if len(pred_idx) == 0:
                continue
            contextual_predictions = contextual.predict(pred_idx, pred_sent_words, pred_sent_tags)
            for sent_id, word_id, pred in zip(pred_sent_ids, pred_idx, contextual_predictions):
                preds[sent_id][word_id] = pred
        return preds

    def update_contextual_preds(self, doc, preds):
        """
        Update a flat list of preds with the output of the contextual lemmatizers

        - First, it unflattens the preds based on the lengths of the sentences
        - Then it uses the contextual lemmatizers
        - Finally, it reflattens the preds into the format expected by the caller
        """
        if len(self.contextual_lemmatizers) == 0:
            return preds

        sentence_words = doc.get([TEXT], as_sentences=True)
        sentence_tags = doc.get([UPOS], as_sentences=True)
        sentence_preds = []
        start_index = 0
        for sent in sentence_words:
            end_index = start_index + len(sent)
            sentence_preds.append(preds[start_index:end_index])
            start_index += len(sent)
        preds = self.predict_contextual(sentence_words, sentence_tags, sentence_preds)
        preds = [lemma for sentence in preds for lemma in sentence]
        return preds

    def update_lr(self, new_lr):
        utils.change_lr(self.optimizer, new_lr)

    def _lookup(self, w, pos):
        """
        Core dict lookup: composite (pos+word) first, then pos-independent fallback.
        Returns None if the word is not in either dict.

        Uses explicit `is not None` checks so that a stored value of None
        (e.g. from a corrupt entry) does not silently fall through to the
        fallback or be treated as a hit.
        """
        pos_entries = self.pos_dict.get(pos)
        if pos_entries is not None:
            lemma = pos_entries.get(w)
            if lemma is not None:
                return lemma
        fallback = self.pos_dict.get(_POS_INDEPENDENT)
        if fallback is not None:
            return fallback.get(w)
        return None

    def train_dict(self, triples, update_word_dict=True):
        """
        Train the dict lemmatizer given training (word, pos, lemma) triples.

        update_word_dict controls whether the pos-independent (_POS_INDEPENDENT)
        entries are updated.  Set to False when adding words at pipeline time
        where tags may be limited.

        Mirrors the original priority logic: most-frequent mapping wins, and
        composite (word+pos) entries are only stored when they differ from the
        most-frequent mapping for that specific (word, pos) pair as recorded in
        pos_independent.
        """
        ctr = Counter()
        ctr.update([(p[0], p[1], p[2]) for p in triples])

        # Build a secondary counter over (w, pos) to find the most frequent
        # lemma per (word, pos) pair — needed to correctly handle cases where
        # the same (w, pos) appears with multiple lemmas across triples.
        wp_ctr = Counter()
        wp_ctr.update([(p[0], p[1]) for p in triples])

        pos_independent = self.pos_dict.setdefault(_POS_INDEPENDENT, {})

        # Track which (w, pos) pairs we have already assigned a composite entry,
        # so that only the most-frequent mapping (first seen in most_common) wins.
        seen_wp = set()

        for (w, pos, l), _ in ctr.most_common():
            # pos-independent fallback: most-frequent mapping for this word wins
            if update_word_dict and w not in pos_independent:
                pos_independent[w] = l
            # composite entry: only the most-frequent lemma for (w, pos) is kept,
            # and only when it genuinely differs from the pos-independent fallback
            if (w, pos) not in seen_wp:
                seen_wp.add((w, pos))
                pos_entries = self.pos_dict.setdefault(pos, {})
                if pos_independent.get(w) != l:
                    pos_entries[w] = l

    def predict_dict(self, pairs):
        """ Predict a list of lemmas using the dict model given (word, pos) pairs. """
        lemmas = []
        for w, pos in pairs:
            if self.caseless:
                w = w.lower()
            lemma = self._lookup(w, pos)
            lemmas.append(lemma if lemma is not None else w)
        return lemmas

    def skip_seq2seq(self, pairs):
        """ Determine if we can skip the seq2seq module when ensembling with the frequency lexicon. """
        skip = []
        for w, pos in pairs:
            if self.caseless:
                w = w.lower()
            skip.append(self._lookup(w, pos) is not None)
        return skip

    def ensemble(self, pairs, other_preds):
        """ Ensemble the dict with statistical model predictions. """
        assert len(pairs) == len(other_preds)
        lemmas = []
        for (w, pos), pred in zip(pairs, other_preds):
            if self.caseless:
                w = w.lower()
            lemma = self._lookup(w, pos)
            if lemma is None:
                lemma = pred
            if lemma is None:
                lemma = w
            lemmas.append(lemma)
        return lemmas

    def save(self, filename, skip_modules=True):
        model_state = None
        if self.model is not None:
            model_state = self.model.state_dict()
            # skip saving modules like the pretrained charlm
            if skip_modules:
                skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
                for k in skipped:
                    del model_state[k]
        params = {
            'model': model_state,
            'dicts': _pack_pos_dict(self.pos_dict),
            'dicts_version': _DICTS_VERSION_POS,
            'vocab': self.vocab.state_dict(),
            'config': self.args,
            'contextual': [],
        }
        for contextual in self.contextual_lemmatizers:
            params['contextual'].append(contextual.get_save_dict())
        save_dir = os.path.split(filename)[0]
        if save_dir:
            os.makedirs(os.path.split(filename)[0], exist_ok=True)
        torch.save(params, filename, _use_new_zipfile_serialization=False)
        logger.info("Model saved to {}".format(filename))

    def load(self, filename, args, foundation_cache, lemma_classifier_args=None):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        if args is not None:
            self.args['charlm_forward_file'] = args.get('charlm_forward_file', self.args['charlm_forward_file'])
            self.args['charlm_backward_file'] = args.get('charlm_backward_file', self.args['charlm_backward_file'])

        dicts_version = checkpoint.get('dicts_version', _DICTS_VERSION_LEGACY)
        if dicts_version == _DICTS_VERSION_POS:
            self.pos_dict = _unpack_pos_dict(checkpoint['dicts'])
        else:
            # legacy format: (word_dict, composite_dict) tuple
            logger.debug("Loading legacy dict format (version %d), converting to pos_dict", dicts_version)
            word_dict, composite_dict = checkpoint['dicts']
            self.pos_dict = _legacy_dicts_to_pos_dict(word_dict, composite_dict)

        if not self.args['dict_only']:
            self.model = self.build_seq2seq(self.args, None, foundation_cache)
            # could remove strict=False after rebuilding all models,
            # or could switch to 1.6.0 torch with the buffer in seq2seq persistent=False
            self.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            self.model = None
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.contextual_lemmatizers = []
        for contextual in checkpoint.get('contextual', []):
            self.contextual_lemmatizers.append(LemmaClassifier.from_checkpoint(contextual, args=lemma_classifier_args))
