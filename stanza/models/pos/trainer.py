"""
A trainer class to handle training and testing of models.
"""

import sys
import logging
import torch
from torch import nn

from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.common import utils, loss
from stanza.models.common.foundation_cache import load_bert, load_bert_with_peft, NoTransformerFoundationCache
from stanza.models.common.peft_config import build_peft_wrapper, load_peft_wrapper
from stanza.models.pos.model import Tagger, remap_legacy_state_dict
from stanza.models.pos.tag_columns import tag_columns_from_args, tag_columns_from_config, tag_columns_to_config
from stanza.models.pos.vocab import MultiVocab

logger = logging.getLogger('stanza')

def unpack_batch(batch, device):
    """ Unpack a batch from the data loader.

    Addressed by name rather than by position, as the number of tag
    columns varies, and with it the offsets of the fields after them.
    """
    def to_device(b):
        return b.to(device) if b is not None else None

    inputs = [to_device(batch.words), to_device(batch.words_mask),
              to_device(batch.wordchars), to_device(batch.wordchars_mask),
              [to_device(tag) for tag in batch.tags],
              to_device(batch.pretrained)]
    return inputs, batch.orig_idx, batch.word_orig_idx, batch.lens, batch.word_lens, batch.text

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, device=None, foundation_cache=None):
        if model_file is not None:
            # load everything from file
            self.load(model_file, pretrain, args=args, foundation_cache=foundation_cache)
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab

            bert_model, bert_tokenizer = load_bert(self.args['bert_model'])
            peft_name = None
            if self.args['use_peft']:
                # fine tune the bert if we're using peft
                self.args['bert_finetune'] = True
                peft_name = "pos"
                bert_model = build_peft_wrapper(bert_model, self.args, logger, adapter_name=peft_name)

            self.model = Tagger(args, vocab, emb_matrix=pretrain.emb if pretrain is not None else None, share_hid=args['share_hid'], foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=self.args['bert_finetune'], peft_name=peft_name)

        self.model = self.model.to(device)
        self.optimizers = utils.get_split_optimizer(self.args['optim'], self.model, self.args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6, weight_decay=self.args.get('initial_weight_decay', None), bert_learning_rate=self.args.get('bert_learning_rate', 0.0), is_peft=self.args.get("peft", False))

        self.schedulers = {}

        if self.args.get('bert_finetune', None):
            import transformers
            warmup_scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizers["bert_optimizer"],
                # todo late starting?
                0, self.args["max_steps"])
            self.schedulers["bert_scheduler"] = warmup_scheduler

    def update(self, batch, eval=False):
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, tags, pretrained = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, tags, pretrained, word_orig_idx, sentlens, wordlens, text)
        if loss == 0.0:
            return loss

        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])

        for optimizer in self.optimizers.values():
            optimizer.step()
        for scheduler in self.schedulers.values():
            scheduler.step()
        return loss_val

    def predict(self, batch, unsort=True):
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, tags, pretrained = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, tags, pretrained, word_orig_idx, sentlens, wordlens, text)

        # one sequence per output column, in column order.  Columns
        # which aren't written back to the document (an extra tagset
        # has nowhere in CoNLL-U to go) are predicted but dropped here
        seqs = [[self.vocab[column.name].unmap(sent) for sent in pred.tolist()]
                for column, pred in zip(self.model.tag_columns, preds)
                if column.output]

        pred_tokens = [[[seq[i][j] for seq in seqs] for j in range(sentlens[i])] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        # the tag columns are namedtuples of enums in the args, but the
        # model file is read back with weights_only, which will only
        # unpickle builtins, so they are flattened to plain strings here
        # and turned back into columns in load()
        config = dict(self.args)
        config['tag_columns'] = tag_columns_to_config(tag_columns_from_args(self.args))
        params = {
                'model': model_state,
                'vocab': self.vocab.state_dict(),
                'config': config
                }
        if self.args.get('use_peft', False):
            # Hide import so that peft dependency is optional
            from peft import get_peft_model_state_dict
            params["bert_lora"] = get_peft_model_state_dict(self.model.bert_model, adapter_name=self.model.peft_name)

        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.warning(f"Saving failed... {e} continuing anyway.")

    def load(self, filename, pretrain, args=None, foundation_cache=None):
        """
        Load a model from file, with preloaded pretrain embeddings. Here we allow the pretrain to be None or a dummy input,
        and the actual use of pretrain embeddings will depend on the boolean config "pretrain" in the loaded args.
        """
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        self.args['tag_columns'] = tag_columns_from_config(self.args.get('tag_columns'))
        if args is not None: self.args.update(args)

        # preserve old models which were created before transformers were added
        if 'bert_model' not in self.args:
            self.args['bert_model'] = None

        lora_weights = checkpoint.get('bert_lora')
        if lora_weights:
            logger.debug("Found peft weights for POS; loading a peft adapter")
            self.args["use_peft"] = True

        # TODO: refactor this common block of code with NER
        force_bert_saved = False
        peft_name = None
        if self.args.get('use_peft', False):
            force_bert_saved = True
            bert_model, bert_tokenizer, peft_name = load_bert_with_peft(self.args['bert_model'], "pos", foundation_cache)
            bert_model = load_peft_wrapper(bert_model, lora_weights, self.args, logger, peft_name)
            logger.debug("Loaded peft with name %s", peft_name)
        else:
            if any(x.startswith("bert_model.") for x in checkpoint['model'].keys()):
                logger.debug("Model %s has a finetuned transformer.  Not using transformer cache to make sure the finetuned version of the transformer isn't accidentally used elsewhere", filename)
                foundation_cache = NoTransformerFoundationCache(foundation_cache)
                force_bert_saved = True
            bert_model, bert_tokenizer = load_bert(self.args.get('bert_model'), foundation_cache)

        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        # load model
        emb_matrix = None
        if self.args['pretrain'] and pretrain is not None: # we use pretrain only if args['pretrain'] == True and pretrain is not None
            emb_matrix = pretrain.emb
        if any(x.startswith("bert_model.") for x in checkpoint['model'].keys()):
            logger.debug("Model %s has a finetuned transformer.  Not using transformer cache to make sure the finetuned version of the transformer isn't accidentally used elsewhere", filename)
            foundation_cache = NoTransformerFoundationCache(foundation_cache)
        self.model = Tagger(self.args, self.vocab, emb_matrix=emb_matrix, share_hid=self.args['share_hid'], foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=force_bert_saved, peft_name=peft_name)
        incompatible = self.model.load_state_dict(remap_legacy_state_dict(checkpoint['model']), strict=False)
        # strict=False, as save() leaves out the modules in
        # unsaved_modules - the transformer, the pretrain, the charlms -
        # which are stored separately.  Those are the only parameters
        # allowed to be absent, so the same list forgives them here.
        # Anything else means the model does not match its parameters,
        # most likely because the tag columns are not the ones it was
        # trained with, and a head would silently be left at its random
        # initialization
        missing = [x for x in incompatible.missing_keys
                   if x.split('.')[0] not in self.model.unsaved_modules]
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise ValueError("Cannot load the POS model %s: the saved parameters do not match the model.  Tag columns are %s.  Missing parameters: %s  Unexpected parameters: %s" %
                             (filename, self.model.tag_names, missing, unexpected))
