"""
A trainer class to handle training and testing of models.
"""

import copy
import sys
import logging
import torch
from torch import nn

try:
    import transformers
except ImportError:
    pass

from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.common import utils, loss
from stanza.models.common.foundation_cache import load_bert, load_bert_with_peft, NoTransformerFoundationCache
from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from stanza.models.common.peft_config import build_peft_wrapper, load_peft_wrapper
from stanza.models.depparse.model import Parser
from stanza.models.pos.vocab import MultiVocab

logger = logging.getLogger('stanza')

def unpack_batch(batch, device):
    """ Unpack a batch from the data loader. """
    inputs = [b.to(device) if b is not None else None for b in batch[:11]]
    orig_idx = batch[11]
    word_orig_idx = batch[12]
    sentlens = batch[13]
    wordlens = batch[14]
    text = batch[15]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens, text

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None,
                 device=None, foundation_cache=None, ignore_model_config=False, reset_history=False):
        self.global_step = 0
        self.last_best_step = 0
        self.dev_score_history = []

        orig_args = copy.deepcopy(args)
        # whether the training is in primary or secondary stage
        # during FT (loading weights), etc., the training is considered to be in "secondary stage"
        # during this time, we (optionally) use a different set of optimizers than that during "primary stage".
        #
        # Regardless, we use TWO SETS of optimizers; once primary converges, we switch to secondary

        if model_file is not None:
            # load everything from file
            self.load(model_file, pretrain, args, foundation_cache, device)

            if reset_history:
                self.global_step = 0
                self.last_best_step = 0
                self.dev_score_history = []
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab

            bert_model, bert_tokenizer = load_bert(self.args['bert_model'])
            peft_name = None
            if self.args['use_peft']:
                # fine tune the bert if we're using peft
                self.args['bert_finetune'] = True
                peft_name = "depparse"
                bert_model = build_peft_wrapper(bert_model, self.args, logger, adapter_name=peft_name)

            self.model = Parser(args, vocab, emb_matrix=pretrain.emb if pretrain is not None else None, foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=self.args['bert_finetune'], peft_name=peft_name)
            self.model = self.model.to(device)
            self.__init_optim()

        if ignore_model_config:
            self.args = orig_args

        if self.args.get('wandb'):
            import wandb
            # track gradients!
            wandb.watch(self.model, log_freq=4, log="all", log_graph=True)

    def __init_optim(self):
        # TODO: can get rid of args.get when models are rebuilt
        if (self.args.get("second_stage", False) and self.args.get('second_optim')):
            self.optimizer = utils.get_split_optimizer(self.args['second_optim'], self.model,
                                                       self.args['second_lr'], betas=(0.9, self.args['beta2']), eps=1e-6,
                                                       bert_learning_rate=self.args.get('second_bert_learning_rate', 0.0),
                                                       is_peft=self.args.get('use_peft', False),
                                                       bert_finetune_layers=self.args.get('bert_finetune_layers', None))
        else:
            self.optimizer = utils.get_split_optimizer(self.args['optim'], self.model,
                                                       self.args['lr'], betas=(0.9, self.args['beta2']),
                                                       eps=1e-6, bert_learning_rate=self.args.get('bert_learning_rate', 0.0),
                                                       weight_decay=self.args.get('weight_decay', None),
                                                       bert_weight_decay=self.args.get('bert_weight_decay', 0.0),
                                                       is_peft=self.args.get('use_peft', False),
                                                       bert_finetune_layers=self.args.get('bert_finetune_layers', None))
        self.scheduler = {}
        if self.args.get("second_stage", False) and self.args.get('second_optim'):
            if self.args.get('second_warmup_steps', None):
                for name, optimizer in self.optimizer.items():
                    name = name + "_scheduler"
                    warmup_scheduler = transformers.get_constant_schedule_with_warmup(optimizer, self.args['second_warmup_steps'])
                    self.scheduler[name] = warmup_scheduler
        else:
            if "bert_optimizer" in self.optimizer:
                zero_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer["bert_optimizer"], factor=0, total_iters=self.args['bert_start_finetuning'])
                warmup_scheduler = transformers.get_constant_schedule_with_warmup(
                    self.optimizer["bert_optimizer"],
                    self.args['bert_warmup_steps'])
                self.scheduler["bert_scheduler"] = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer["bert_optimizer"],
                    schedulers=[zero_scheduler, warmup_scheduler],
                    milestones=[self.args['bert_start_finetuning']])

    def update(self, batch, eval=False):
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            for opt in self.optimizer.values():
                opt.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        for opt in self.optimizer.values():
            opt.step()
        for scheduler in self.scheduler.values():
            scheduler.step()
        return loss_val

    def predict(self, batch, unsort=True):
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], sentlens)] # remove attachment for the root
        deprel_seqs = [self.vocab['deprel'].unmap([preds[1][i][j+1][h] for j, h in enumerate(hs)]) for i, hs in enumerate(head_seqs)]

        pred_tokens = [[[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(sentlens[i]-1)] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

    def save(self, filename, skip_modules=True, save_optimizer=False):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
                'model': model_state,
                'vocab': self.vocab.state_dict(),
                'config': self.args,
                'global_step': self.global_step,
                'last_best_step': self.last_best_step,
                'dev_score_history': self.dev_score_history,
                }
        if self.args.get('use_peft', False):
            # Hide import so that peft dependency is optional
            from peft import get_peft_model_state_dict
            params["bert_lora"] = get_peft_model_state_dict(self.model.bert_model, adapter_name=self.model.peft_name)

        if save_optimizer and self.optimizer is not None:
            params['optimizer_state_dict'] = {k: opt.state_dict() for k, opt in self.optimizer.items()}
            params['scheduler_state_dict'] = {k: scheduler.state_dict() for k, scheduler in self.scheduler.items()}

        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except BaseException:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename, pretrain, args=None, foundation_cache=None, device=None):
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
        if args is not None: self.args.update(args)

        # preserve old models which were created before transformers were added
        if 'bert_model' not in self.args:
            self.args['bert_model'] = None

        lora_weights = checkpoint.get('bert_lora')
        if lora_weights:
            logger.debug("Found peft weights for depparse; loading a peft adapter")
            self.args["use_peft"] = True

        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        # load model
        emb_matrix = None
        if self.args['pretrain'] and pretrain is not None: # we use pretrain only if args['pretrain'] == True and pretrain is not None
            emb_matrix = pretrain.emb

        # TODO: refactor this common block of code with NER
        force_bert_saved = False
        peft_name = None
        if self.args.get('use_peft', False):
            force_bert_saved = True
            bert_model, bert_tokenizer, peft_name = load_bert_with_peft(self.args['bert_model'], "depparse", foundation_cache)
            bert_model = load_peft_wrapper(bert_model, lora_weights, self.args, logger, peft_name)
            logger.debug("Loaded peft with name %s", peft_name)
        else:
            if any(x.startswith("bert_model.") for x in checkpoint['model'].keys()):
                logger.debug("Model %s has a finetuned transformer.  Not using transformer cache to make sure the finetuned version of the transformer isn't accidentally used elsewhere", filename)
                foundation_cache = NoTransformerFoundationCache(foundation_cache)
                force_bert_saved = True
            bert_model, bert_tokenizer = load_bert(self.args.get('bert_model'), foundation_cache)

        self.model = Parser(self.args, self.vocab, emb_matrix=emb_matrix, foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=force_bert_saved, peft_name=peft_name)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        if device is not None:
            self.model = self.model.to(device)

        self.__init_optim()
        optim_state_dict = checkpoint.get("optimizer_state_dict")
        if optim_state_dict:
            for k, state in optim_state_dict.items():
                self.optimizer[k].load_state_dict(state)

        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        if scheduler_state_dict:
            for k, state in scheduler_state_dict.items():
                self.scheduler[k].load_state_dict(state)

        self.global_step = checkpoint.get("global_step", 0)
        self.last_best_step = checkpoint.get("last_best_step", 0)
        self.dev_score_history = checkpoint.get("dev_score_history", list())
