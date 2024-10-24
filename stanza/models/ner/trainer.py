"""
A trainer class to handle training and testing of models.
"""

import sys
import logging
import torch
from torch import nn

from stanza.models.common.foundation_cache import NoTransformerFoundationCache, load_bert, load_bert_with_peft
from stanza.models.common.peft_config import build_peft_wrapper, load_peft_wrapper
from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.common.vocab import VOCAB_PREFIX, VOCAB_PREFIX_SIZE
from stanza.models.common import utils, loss
from stanza.models.ner.model import NERTagger
from stanza.models.ner.vocab import MultiVocab
from stanza.models.common.crf import viterbi_decode


logger = logging.getLogger('stanza')

def unpack_batch(batch, device):
    """ Unpack a batch from the data loader. """
    inputs = [batch[0]]
    inputs += [b.to(device) if b is not None else None for b in batch[1:5]]
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
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, device=None,
                 train_classifier_only=False, foundation_cache=None, second_optim=False):
        if model_file is not None:
            # load everything from file
            self.load(model_file, pretrain, args, foundation_cache)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            bert_model, bert_tokenizer = load_bert(self.args['bert_model'])
            peft_name = None
            if self.args['use_peft']:
                # fine tune the bert if we're using peft
                self.args['bert_finetune'] = True
                peft_name = "ner"
                # peft the lovely model
                bert_model = build_peft_wrapper(bert_model, self.args, logger, adapter_name=peft_name)

            self.model = NERTagger(args, vocab, emb_matrix=pretrain.emb, foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=self.args['bert_finetune'], peft_name=peft_name)

            # IMPORTANT: gradient checkpointing BREAKS peft if applied before
            # 1. Apply PEFT FIRST (looksie! it's above this line)
            # 2. Run gradient checkpointing
            # https://github.com/huggingface/peft/issues/742
            if self.args.get("gradient_checkpointing", False) and self.args.get("bert_finetune", False):
                self.model.bert_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


        # if this wasn't set anywhere, we use a default of the 0th tagset
        # we don't set this as a default in the options so that
        # we can distinguish "intentionally set to 0" and "not set at all"
        if self.args.get('predict_tagset', None) is None:
            self.args['predict_tagset'] = 0

        if train_classifier_only:
            logger.info('Disabling gradient for non-classifier layers')
            exclude = ['tag_clf', 'crit']
            for pname, p in self.model.named_parameters():
                if pname.split('.')[0] not in exclude:
                    p.requires_grad = False
        self.model = self.model.to(device)
        if not second_optim:
            self.optimizer = utils.get_optimizer(self.args['optim'], self.model, self.args['lr'], momentum=self.args['momentum'], bert_learning_rate=self.args.get('bert_learning_rate', 0.0), is_peft=self.args.get("use_peft"))
        else:
            self.optimizer = utils.get_optimizer(self.args['second_optim'], self.model, self.args['second_lr'], momentum=self.args['momentum'], bert_learning_rate=self.args.get('second_bert_learning_rate', 0.0), is_peft=self.args.get("use_peft"))

    def update(self, batch, eval=False):
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(batch, device)
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
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(batch, device)
        word, wordchars, wordchars_mask, chars, tags = inputs

        self.model.eval()
        #batch_size = word.size(0)
        _, logits, trans = self.model(word, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx)

        # decode
        # TODO: might need to decode multiple columns of output for
        # models with multiple layers
        trans = [x.data.cpu().numpy() for x in trans]
        logits = [x.data.cpu().numpy() for x in logits]
        batch_size = logits[0].shape[0]
        if any(x.shape[0] != batch_size for x in logits):
            raise AssertionError("Expected all of the logits to have the same size")
        tag_seqs = []
        predict_tagset = self.args['predict_tagset']
        for i in range(batch_size):
            # for each tag column in the output, decode the tag assignments
            tags = [viterbi_decode(x[i, :sentlens[i]], y)[0] for x, y in zip(logits, trans)]
            # TODO: this is to patch that the model can sometimes predict < "O"
            tags = [[x if x >= VOCAB_PREFIX_SIZE else VOCAB_PREFIX_SIZE for x in y] for y in tags]
            # that gives us N lists of |sent| tags, whereas we want |sent| lists of N tags
            tags = list(zip(*tags))
            # now unmap that to the tags in the vocab
            tags = self.vocab['tag'].unmap(tags)
            # for now, allow either TagVocab or CompositeVocab
            # TODO: we might want to return all of the predictions
            # rather than a single column
            tags = [x[predict_tagset] if isinstance(x, list) else x for x in tags]
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

        if self.args["use_peft"]:
            from peft import get_peft_model_state_dict
            params["bert_lora"] = get_peft_model_state_dict(self.model.bert_model, adapter_name=self.model.peft_name)
        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename, pretrain=None, args=None, foundation_cache=None):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        if args: self.args.update(args)
        # if predict_tagset was not explicitly set in the args,
        # we use the value the model was trained with
        for keep_arg in ('predict_tagset', 'train_scheme', 'scheme'):
            if self.args.get(keep_arg, None) is None:
                self.args[keep_arg] = checkpoint['config'].get(keep_arg, None)

        lora_weights = checkpoint.get('bert_lora')
        if lora_weights:
            logger.debug("Found peft weights for NER; loading a peft adapter")
            self.args["use_peft"] = True

        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])

        emb_matrix=None
        if pretrain is not None:
            emb_matrix = pretrain.emb

        force_bert_saved = False
        peft_name = None
        if self.args.get('use_peft', False):
            force_bert_saved = True
            bert_model, bert_tokenizer, peft_name = load_bert_with_peft(self.args['bert_model'], "ner", foundation_cache)
            bert_model = load_peft_wrapper(bert_model, lora_weights, self.args, logger, peft_name)
            logger.debug("Loaded peft with name %s", peft_name)
        else:
            if any(x.startswith("bert_model.") for x in checkpoint['model'].keys()):
                logger.debug("Model %s has a finetuned transformer.  Not using transformer cache to make sure the finetuned version of the transformer isn't accidentally used elsewhere", filename)
                foundation_cache = NoTransformerFoundationCache(foundation_cache)
                force_bert_saved = True
            bert_model, bert_tokenizer = load_bert(self.args.get('bert_model'), foundation_cache)

        if any(x.startswith("crit.") for x in checkpoint['model'].keys()):
            logger.debug("Old model format detected.  Updating to the new format with one column of tags")
            checkpoint['model']['crits.0._transitions'] = checkpoint['model'].pop('crit._transitions')
            checkpoint['model']['tag_clfs.0.weight'] = checkpoint['model'].pop('tag_clf.weight')
            checkpoint['model']['tag_clfs.0.bias'] = checkpoint['model'].pop('tag_clf.bias')
        self.model = NERTagger(self.args, self.vocab, emb_matrix=emb_matrix, foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=force_bert_saved, peft_name=peft_name)
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
        for tag in self.vocab['tag'].items(0):
            if tag in VOCAB_PREFIX:
                continue
            if tag == 'O':
                continue
            if len(tag) > 2 and tag[:2] in ('S-', 'B-', 'I-', 'E-'):
                tag = tag[2:]
            tags.add(tag)
        return sorted(tags)
