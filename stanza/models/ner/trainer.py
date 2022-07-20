"""
A trainer class to handle training and testing of models.
"""

import sys
import logging
import torch
from torch import nn

from stanza.models.common.foundation_cache import load_bert
from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.common.vocab import VOCAB_PREFIX
from stanza.models.common import utils, loss
from stanza.models.ner.model import NERTagger
from stanza.models.ner.vocab import MultiVocab
from stanza.models.common.crf import viterbi_decode

logger = logging.getLogger('stanza')

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [batch[0]]
        inputs += [b.cuda() if b is not None else None for b in batch[1:5]]
    else:
        inputs = batch[:5]
    orig_idx = batch[5]
    word_orig_idx = batch[6]
    char_orig_idx = batch[7]
    sentlens = batch[8]
    wordlens = batch[9]
    charlens = batch[10]
    charoffsets = batch[11]
    return inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets

def fix_singleton_tags(tags):
    """
    If there are any singleton B- or E- tags, convert them to S-
    """
    new_tags = list(tags)
    # first update all I- tags at the start or end of sequence to B- or E- as appropriate
    for idx, tag in enumerate(new_tags):
        if (tag.startswith("I-") and
            (idx == len(new_tags) - 1 or
             (new_tags[idx+1] != "I-" + tag[2:] and new_tags[idx+1] != "E-" + tag[2:]))):
            new_tags[idx] = "E-" + tag[2:]
        if (tag.startswith("I-") and
            (idx == 0 or
             (new_tags[idx-1] != "B-" + tag[2:] and new_tags[idx-1] != "I-" + tag[2:]))):
            new_tags[idx] = "B-" + tag[2:]
    # now make another pass through the data to update any singleton tags,
    # including ones which were turned into singletons by the previous operation
    for idx, tag in enumerate(new_tags):
        if (tag.startswith("B-") and
            (idx == len(new_tags) - 1 or
             (new_tags[idx+1] != "I-" + tag[2:] and new_tags[idx+1] != "E-" + tag[2:]))):
            new_tags[idx] = "S-" + tag[2:]
        if (tag.startswith("E-") and
            (idx == 0 or
             (new_tags[idx-1] != "B-" + tag[2:] and new_tags[idx-1] != "I-" + tag[2:]))):
            new_tags[idx] = "S-" + tag[2:]
    return new_tags

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False,
                 train_classifier_only=False, foundation_cache=None):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(model_file, pretrain, args, foundation_cache)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.bert_model, self.bert_tokenizer = load_bert(args['bert_model'], foundation_cache)
            self.model = NERTagger(args, vocab, emb_matrix=pretrain.emb, bert_model = self.bert_model, bert_tokenizer = self.bert_tokenizer, use_cuda = self.use_cuda)

        if train_classifier_only:
            logger.info('Disabling gradient for non-classifier layers')
            exclude = ['tag_clf', 'crit']
            for pname, p in self.model.named_parameters():
                if pname.split('.')[0] not in exclude:
                    p.requires_grad = False
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], momentum=self.args['momentum'])

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(batch, self.use_cuda)
        word, wordchars, wordchars_mask, chars, tags = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _, _ = self.model(word, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(batch, self.use_cuda)
        word, wordchars, wordchars_mask, chars, tags = inputs

        self.model.eval()
        #batch_size = word.size(0)
        _, logits, trans = self.model(word, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx)

        # decode
        trans = trans.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :sentlens[i]], trans)
            tags = self.vocab['tag'].unmap(tags)
            tags = fix_singleton_tags(tags)
            tag_seqs += [tags]

        if unsort:
            tag_seqs = utils.unsort(tag_seqs, orig_idx)
        return tag_seqs

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
                'model': model_state,
                'vocab': self.vocab.state_dict(),
                'config': self.args
                }
        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename, pretrain=None, args=None, foundation_cache=None):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        if args: self.args.update(args)
        self.bert_model, self.bert_tokenizer = load_bert(self.args.get('bert_model', None), foundation_cache)
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])

        emb_matrix=None
        if pretrain is not None:
            emb_matrix = pretrain.emb

        self.model = NERTagger(self.args, self.vocab, emb_matrix=emb_matrix, bert_model=self.bert_model, bert_tokenizer=self.bert_tokenizer, use_cuda=self.use_cuda)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # there is a possible issue with the delta embeddings.
        # specifically, with older models trained without the delta
        # embedding matrix
        # if those models have been trained with the embedding
        # modifications saved as part of the base embedding,
        # we need to resave the model with the updated embedding
        # otherwise the resulting model will be broken
        if 'delta' not in self.model.vocab and 'word_emb.weight' in checkpoint['model'].keys() and 'word_emb' in self.model.unsaved_modules:
            logger.debug("Removing word_emb from unsaved_modules so that resaving %s will keep the saved embedding", filename)
            self.model.unsaved_modules.remove('word_emb')

    def get_known_tags(self):
        """
        Return the tags known by this model

        Removes the S-, B-, etc, and does not include O
        """
        tags = set()
        for tag in self.vocab['tag']:
            if tag in VOCAB_PREFIX:
                continue
            if tag == 'O':
                continue
            if len(tag) > 2 and tag[:2] in ('S-', 'B-', 'I-', 'E-'):
                tag = tag[2:]
            tags.add(tag)
        return sorted(tags)
