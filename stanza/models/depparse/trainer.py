"""
A trainer class to handle training and testing of models.
"""

from abc import ABC, abstractmethod
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
from stanza.models.common.peft_config import build_peft_wrapper, load_peft_wrapper
from stanza.models.common.warmup_plateau_scheduler import WarmupThenPlateauScheduler
from stanza.models.depparse.model import EnsembleGraphParser, GraphParser
from stanza.models.depparse.transition.model import SubtreeCombination, EnsembleTransitionParser, TransitionParser
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

# TODO: there was an ignore_model_config option which is mostly
# replaced by having the load() method use the passed-in args.
# double check that all of that works
class Trainer(BaseTrainer, ABC):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, pretrain=None, model=None,
                 device=None, foundation_cache=None, build_optimizer=True):
        self.global_step = 0
        self.last_best_step = 0
        self.dev_score_history = []
        self.model_name = None

        # whether the training is in primary or secondary stage
        # during FT (loading weights), etc., the training is considered to be in "secondary stage"
        # during this time, we (optionally) use a different set of optimizers than that during "primary stage".
        #
        # Regardless, we use TWO SETS of optimizers; once primary converges, we switch to secondary

        if model is not None:
            # load everything from file
            self.model = model
            self.args = args
            self.vocab = vocab
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab

            bert_model, bert_tokenizer = load_bert(self.args['bert_model'], enable_gradient_checkpointing=args['enable_gradient_checkpointing'])
            peft_name = None
            if self.args['use_peft']:
                # fine tune the bert if we're using peft
                self.args['bert_finetune'] = True
                peft_name = "depparse"
                bert_model = build_peft_wrapper(bert_model, self.args, logger, adapter_name=peft_name)

            self.model = self.build_model(args, vocab, emb_matrix=pretrain.emb if pretrain is not None else None, foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=self.args['bert_finetune'], peft_name=peft_name)
            self.model = self.model.to(device)

        self.optimizer = None
        self.scheduler = None
        if build_optimizer:
            self.__init_optim()

        if self.args.get('wandb'):
            import wandb
            # track gradients!
            wandb.watch(self.model, log_freq=4, log="all", log_graph=True)

    @staticmethod
    def build_model():
        """ Create a model for this particular type of parser """
        raise NotImplementedError()

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
        for name, optimizer in self.optimizer.items():
            name = name + "_scheduler"
            if self.args.get("second_stage", False) and self.args.get('second_optim'):
                num_freeze_steps = 0
                num_warmup_steps = self.args.get('second_warmup_steps', 0)
            else:
                num_freeze_steps = 0
                num_warmup_steps = 0
                if name.startswith("bert_") or name.startswith("peft_"):
                    num_freeze_steps = self.args.get('bert_start_finetuning', 0)
                    num_warmup_steps = self.args.get('bert_warmup_steps', 0)
            patience = self.args.get('plateau_steps', None) if self.args.get('use_plateau') else None
            decay = self.args.get('plateau_decay', 0.9)
            logger.debug("Building scheduler %s with num_freeze_steps %d, num_warmup_steps %d, decay factor %f, patience %s",
                         name, num_freeze_steps, num_warmup_steps, decay, patience)
            warmup_scheduler = WarmupThenPlateauScheduler(optimizer,
                                                          num_freeze_steps = num_freeze_steps,
                                                          num_warmup_steps = num_warmup_steps,
                                                          reset_optimizer_on_unfreeze=True,
                                                          mode = "max",   # we are passing in F1 scores
                                                          factor = decay,
                                                          patience = patience)
            self.scheduler[name] = warmup_scheduler
        self.bert_finetuning = any(x.startswith("bert") or x.startswith("peft") for x in self.optimizer)
        self.model.bert_finetuning = self.bert_finetuning
        logger.debug("Bert finetuning during this training portion: %s", self.model.bert_finetuning)

    def update(self, batch, eval=False):
        device = self.model.get_device()
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            for opt in self.optimizer.values():
                opt.zero_grad()
        loss, batch_stats = self.model.loss(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        for opt in self.optimizer.values():
            opt.step()
        return loss_val, batch_stats

    def predict(self, batch, unsort=True):
        device = self.model.get_device()
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        self.model.eval()
        pred_tokens = self.model.predict(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        if self.args.get('reversed', False):
            pred_tokens = self.reverse_predictions(pred_tokens)
        return pred_tokens

    def reverse_predictions(self, pred_tokens):
        new_predictions = []
        for sentence in pred_tokens:
            new_sentence = []
            for token in sentence[::-1]:
                if token[0] == 0:
                    new_sentence.append(token)
                else:
                    new_sentence.append((len(sentence) + 1 - token[0], token[1]))
            new_predictions.append(new_sentence)
        return new_predictions

    def save(self, filename, skip_modules=True, save_optimizer=False):
        model_state = self.model.get_params(skip_modules)
        config = dict(self.args)
        # sanitize enums for torch.load(weights_only=True)
        if 'transition_subtree_combination' in config:
            config['transition_subtree_combination'] = config['transition_subtree_combination'].name
        if isinstance(self.model, GraphParser):
            model_type = "graph"
        elif isinstance(self.model, TransitionParser):
            model_type = "transition"
        elif isinstance(self.model, EnsembleGraphParser):
            model_type = "ensemble_graph"
        elif isinstance(self.model, EnsembleTransitionParser):
            model_type = "ensemble_transition"
        else:
            raise ValueError("Unknown model type: %s" % type(self.model))
        # remove the gradient checkpointing arg so that when we reload,
        # the model isn't trying to gradient checkpoint in a pipeline
        if "enable_gradient_checkpointing" in config:
            config.pop("enable_gradient_checkpointing")
        params = {
            'model': model_state,
            'vocab': self.vocab.state_dict(),
            'config': config,
            'global_step': self.global_step,
            'last_best_step': self.last_best_step,
            'dev_score_history': self.dev_score_history,
            'model_type': model_type,
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
        except BaseException as e:
            logger.warning("Saving failed... continuing anyway.  Error was: %s" % e)

    @staticmethod
    def load(filename, pretrain, args=None, foundation_cache=None, device=None, reset_history=False):
        """
        Load a model from file, with preloaded pretrain embeddings. Here we allow the pretrain to be None or a dummy input,
        and the actual use of pretrain embeddings will depend on the boolean config "pretrain" in the loaded args.
        """
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        return Trainer.load_checkpoint(filename, checkpoint, pretrain, args, foundation_cache, device, reset_history)

    @staticmethod
    def load_checkpoint(model_name, checkpoint, pretrain, args=None, foundation_cache=None, device=None, reset_history=False):
        loaded_args = checkpoint['config']
        # enums were sanitized so that weights_only=True works correctly
        transition_subtree_combination = loaded_args.get('transition_subtree_combination')
        transition_subtree_combination = SubtreeCombination[transition_subtree_combination] if transition_subtree_combination is not None else SubtreeCombination.NONE
        loaded_args['transition_subtree_combination'] = transition_subtree_combination
        if args is not None: loaded_args.update(args)

        model_type = checkpoint.get("model_type")
        if not model_type:
            if 'output_basic.weight' in checkpoint['model']:
                model_type = "transition"
            else:
                model_type = "graph"

        vocab = MultiVocab.load_state_dict(checkpoint['vocab'])

        if model_type in ('ensemble_graph', 'ensemble_transition'):
            models = []
            for model_idx, (sub_params, sub_args) in enumerate(zip(checkpoint['model']['params'], checkpoint['model']['args'])):
                # TODO: refactor
                sub_checkpoint = {
                    'model': sub_params,
                    'vocab': checkpoint['vocab'],
                    'config': sub_args,
                    'model_type': 'graph' if model_type == 'ensemble_graph' else 'transition',
                }
                sub_trainer = Trainer.load_checkpoint("%s-%d" % (model_name, model_idx), sub_checkpoint, pretrain, args, foundation_cache, device, reset_history)
                models.append(sub_trainer.model)
            if model_type == 'ensemble_graph':
                model = EnsembleGraphParser(loaded_args, vocab, models)
            else:
                model = EnsembleTransitionParser(loaded_args, vocab, models)
            trainer = Trainer(args=loaded_args, vocab=vocab, model=model, build_optimizer=False)
        else:
            # preserve old models which were created before transformers were added
            if 'bert_model' not in loaded_args:
                loaded_args['bert_model'] = None

            lora_weights = checkpoint.get('bert_lora')
            if lora_weights:
                logger.debug("Found peft weights for depparse; loading a peft adapter")
                loaded_args["use_peft"] = True

            # the loaded_args should not have been saved with this value
            # (it gets removed in save())
            # but the passed in args from the main program might have it,
            # if someone deliberately set it while training
            enable_gradient_checkpointing = loaded_args.get('enable_gradient_checkpointing')
            # load model
            emb_matrix = None
            if loaded_args['pretrain'] and pretrain is not None: # we use pretrain only if args['pretrain'] == True and pretrain is not None
                emb_matrix = pretrain.emb

            # TODO: refactor this common block of code with NER
            force_bert_saved = False
            peft_name = None
            if loaded_args.get('use_peft', False):
                force_bert_saved = True
                bert_model, bert_tokenizer, peft_name = load_bert_with_peft(loaded_args['bert_model'], "depparse", foundation_cache, enable_gradient_checkpointing=enable_gradient_checkpointing)
                bert_model = load_peft_wrapper(bert_model, lora_weights, loaded_args, logger, peft_name)
                logger.debug("Loaded peft with name %s", peft_name)
            else:
                if any(x.startswith("bert_model.") for x in checkpoint['model'].keys()):
                    logger.debug("Model %s has a finetuned transformer.  Not using transformer cache to make sure the finetuned version of the transformer isn't accidentally used elsewhere", model_name)
                    foundation_cache = NoTransformerFoundationCache(foundation_cache)
                    force_bert_saved = True
                bert_model, bert_tokenizer = load_bert(loaded_args.get('bert_model'), foundation_cache, enable_gradient_checkpointing=enable_gradient_checkpointing)

            if 'output_basic.weight' in checkpoint['model']:
                model = TransitionTrainer.build_model(loaded_args, vocab, emb_matrix=emb_matrix, foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=force_bert_saved, peft_name=peft_name)
            else:
                model = GraphTrainer.build_model(loaded_args, vocab, emb_matrix=emb_matrix, foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=force_bert_saved, peft_name=peft_name)
            model.load_params(checkpoint['model'])
            if device is not None:
                model = model.to(device)
            if 'output_basic.weight' in checkpoint['model']:
                trainer = TransitionTrainer(args=loaded_args, vocab=vocab, pretrain=pretrain, model=model, device=device, foundation_cache=foundation_cache)
            else:
                trainer = GraphTrainer(args=loaded_args, vocab=vocab, pretrain=pretrain, model=model, device=device, foundation_cache=foundation_cache)

        optim_state_dict = checkpoint.get("optimizer_state_dict")
        if optim_state_dict:
            for k, state in optim_state_dict.items():
                trainer.optimizer[k].load_state_dict(state)

        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        if scheduler_state_dict:
            for k, state in scheduler_state_dict.items():
                trainer.scheduler[k].load_state_dict(state)

        if reset_history:
            trainer.global_step = 0
            trainer.last_best_step = 0
            trainer.dev_score_history = []
        else:
            trainer.global_step = checkpoint.get("global_step", 0)
            trainer.last_best_step = checkpoint.get("last_best_step", 0)
            trainer.dev_score_history = checkpoint.get("dev_score_history", list())
        trainer.model_name = model_name
        return trainer

class GraphTrainer(Trainer):
    @staticmethod
    def build_model(*args, **kwargs):
        return GraphParser(*args, **kwargs)

class TransitionTrainer(Trainer):
    @staticmethod
    def build_model(*args, **kwargs):
        return TransitionParser(*args, **kwargs)
