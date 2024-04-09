"""
This file includes a variety of methods needed to train new
constituency parsers.  It also includes a method to load an
already-trained parser.

See the `train` method for the code block which starts from
  raw treebank and returns a new parser.
`evaluate` reads a treebank and gives a score for those trees.
"""

import copy
import logging
import os

import torch

from stanza.models.common.foundation_cache import load_bert, load_bert_with_peft, load_charlm, load_pretrain, NoTransformerFoundationCache
from stanza.models.common.peft_config import build_peft_wrapper, load_peft_wrapper
from stanza.models.constituency.lstm_model import LSTMModel
from stanza.models.constituency.utils import build_optimizer, build_scheduler
# TODO: could put find_wordvec_pretrain, choose_charlm, etc in a more central place if it becomes widely used
from stanza.utils.training.common import find_wordvec_pretrain, choose_charlm, find_charlm_file
from stanza.resources.default_packages import default_charlms, default_pretrains

logger = logging.getLogger('stanza')
tlogger = logging.getLogger('stanza.constituency.trainer')

class Trainer:
    """
    Stores a constituency model and its optimizer

    Not inheriting from common/trainer.py because there's no concept of change_lr (yet?)
    """
    def __init__(self, model, optimizer=None, scheduler=None, epochs_trained=0, batches_trained=0, best_f1=0.0, best_epoch=0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # keeping track of the epochs trained will be useful
        # for adjusting the learning scheme
        self.epochs_trained = epochs_trained
        self.batches_trained = batches_trained
        self.best_f1 = best_f1
        self.best_epoch = best_epoch

    def save(self, filename, save_optimizer=True):
        """
        Save the model (and by default the optimizer) to the given path
        """
        params = self.model.get_params()
        checkpoint = {
            'params': params,
            'epochs_trained': self.epochs_trained,
            'batches_trained': self.batches_trained,
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
        }
        if self.model.args.get('use_peft', False):
            # Hide import so that peft dependency is optional
            from peft import get_peft_model_state_dict
            checkpoint["bert_lora"] = get_peft_model_state_dict(self.model.bert_model, adapter_name=self.model.peft_name)
        if save_optimizer and self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, filename, _use_new_zipfile_serialization=False)
        logger.info("Model saved to %s", filename)

    def log_norms(self):
        self.model.log_norms()

    def log_shapes(self):
        self.model.log_shapes()

    @property
    def transitions(self):
        return self.model.transitions

    @property
    def root_labels(self):
        return self.model.root_labels

    @property
    def device(self):
        return next(self.model.parameters()).device

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    @staticmethod
    def find_and_load_pretrain(saved_args, foundation_cache):
        if 'wordvec_pretrain_file' not in saved_args:
            return None
        if os.path.exists(saved_args['wordvec_pretrain_file']):
            return load_pretrain(saved_args['wordvec_pretrain_file'], foundation_cache)
        logger.info("Unable to find pretrain in %s  Will try to load from the default resources instead", saved_args['wordvec_pretrain_file'])
        language = saved_args['lang']
        wordvec_pretrain = find_wordvec_pretrain(language, default_pretrains)
        return load_pretrain(wordvec_pretrain, foundation_cache)

    @staticmethod
    def find_and_load_charlm(charlm_file, direction, saved_args, foundation_cache):
        try:
            return load_charlm(charlm_file, foundation_cache)
        except FileNotFoundError as e:
            logger.info("Unable to load charlm from %s  Will try to load from the default resources instead", charlm_file)
            language = saved_args['lang']
            dataset = saved_args['shorthand'].split("_")[1]
            charlm = choose_charlm(language, dataset, "default", default_charlms, {})
            charlm_file = find_charlm_file(direction, language, charlm)
            return load_charlm(charlm_file, foundation_cache)

    @staticmethod
    def model_from_params(params, peft_params, args, foundation_cache=None, peft_name=None):
        """
        Build a new model just from the saved params and some extra args

        Refactoring allows other processors to include a constituency parser as a module
        """
        saved_args = dict(params['config'])
        # some parameters which change the structure of a model have
        # to be ignored, or the model will not function when it is
        # reloaded from disk
        if args is None: args = {}
        update_args = copy.deepcopy(args)
        # TODO: pop all the peft args as well
        update_args.pop("bert_hidden_layers", None)
        update_args.pop("constituency_composition", None)
        update_args.pop("constituent_stack", None)
        update_args.pop("num_tree_lstm_layers", None)
        update_args.pop("transition_scheme", None)
        update_args.pop("transition_stack", None)
        update_args.pop("maxout_k", None)
        # if the pretrain or charlms are not specified, don't override the values in the model
        # (if any), since the model won't even work without loading the same charlm
        if 'wordvec_pretrain_file' in update_args and update_args['wordvec_pretrain_file'] is None:
            update_args.pop('wordvec_pretrain_file')
        if 'charlm_forward_file' in update_args and update_args['charlm_forward_file'] is None:
            update_args.pop('charlm_forward_file')
        if 'charlm_backward_file' in update_args and update_args['charlm_backward_file'] is None:
            update_args.pop('charlm_backward_file')
        # we don't pop bert_finetune, with the theory being that if
        # the saved model has bert_finetune==True we can load the bert
        # weights but then not further finetune if bert_finetune==False
        saved_args.update(update_args)

        # TODO: not needed if we rebuild the models
        if saved_args.get("bert_finetune", None) is None:
            saved_args["bert_finetune"] = False
        if saved_args.get("stage1_bert_finetune", None) is None:
            saved_args["stage1_bert_finetune"] = False

        model_type = params['model_type']
        if model_type == 'LSTM':
            pt = Trainer.find_and_load_pretrain(saved_args, foundation_cache)
            if saved_args.get('use_peft', False):
                # if loading a peft model, we first load the base transformer
                # then we load the weights using the saved weights in the file
                if peft_name is None:
                    bert_model, bert_tokenizer, peft_name = load_bert_with_peft(saved_args.get('bert_model', None), "constituency", foundation_cache)
                else:
                    bert_model, bert_tokenizer = load_bert(saved_args.get('bert_model', None), foundation_cache)
                bert_model = load_peft_wrapper(bert_model, peft_params, saved_args, logger, peft_name)
                bert_saved = True
            elif saved_args['bert_finetune'] or saved_args['stage1_bert_finetune'] or any(x.startswith("bert_model.") for x in params['model'].keys()):
                # if bert_finetune is True, don't use the cached model!
                # otherwise, other uses of the cached model will be ruined
                bert_model, bert_tokenizer = load_bert(saved_args.get('bert_model', None))
                bert_saved = True
            else:
                bert_model, bert_tokenizer = load_bert(saved_args.get('bert_model', None), foundation_cache)
                bert_saved = False
            forward_charlm =  Trainer.find_and_load_charlm(saved_args["charlm_forward_file"],  "forward",  saved_args, foundation_cache)
            backward_charlm = Trainer.find_and_load_charlm(saved_args["charlm_backward_file"], "backward", saved_args, foundation_cache)
            model = LSTMModel(pretrain=pt,
                              forward_charlm=forward_charlm,
                              backward_charlm=backward_charlm,
                              bert_model=bert_model,
                              bert_tokenizer=bert_tokenizer,
                              force_bert_saved=bert_saved,
                              peft_name=peft_name,
                              transitions=params['transitions'],
                              constituents=params['constituents'],
                              tags=params['tags'],
                              words=params['words'],
                              rare_words=params['rare_words'],
                              root_labels=params['root_labels'],
                              constituent_opens=params['constituent_opens'],
                              unary_limit=params['unary_limit'],
                              args=saved_args)
        else:
            raise ValueError("Unknown model type {}".format(model_type))
        model.load_state_dict(params['model'], strict=False)
        # model will stay on CPU if device==None
        # can be moved elsewhere later, of course
        model = model.to(args.get('device', None))
        return model

    @staticmethod
    def load(filename, args=None, load_optimizer=False, foundation_cache=None, peft_name=None):
        """
        Load back a model and possibly its optimizer.
        """
        if not os.path.exists(filename):
            if args.get('save_dir', None) is None:
                raise FileNotFoundError("Cannot find model in {} and args['save_dir'] is None".format(filename))
            elif os.path.exists(os.path.join(args['save_dir'], filename)):
                filename = os.path.join(args['save_dir'], filename)
            else:
                raise FileNotFoundError("Cannot find model in {} or in {}".format(filename, os.path.join(args['save_dir'], filename)))
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from %s", filename)
            raise
        logger.debug("Loaded model from %s", filename)

        params = checkpoint['params']
        model = Trainer.model_from_params(params, checkpoint.get('bert_lora', None), args, foundation_cache, peft_name)

        epochs_trained = checkpoint['epochs_trained']
        batches_trained = checkpoint.get('batches_trained', 0)
        best_f1 = checkpoint['best_f1']
        best_epoch = checkpoint['best_epoch']

        if load_optimizer:
            # we use params['config'] here instead of model.args
            # because the args might have a different training
            # mechanism, but in order to reload the optimizer, we need
            # to match the optimizer we build with the one that was
            # used at training time
            build_simple_adadelta = params['config']['multistage'] and epochs_trained < params['config']['epochs'] // 2
            logger.debug("Model loaded was built with multistage %s  epochs_trained %d out of total epochs %d  Building initial Adadelta optimizer: %s", params['config']['multistage'], epochs_trained, params['config']['epochs'], build_simple_adadelta)
            optimizer = build_optimizer(model.args, model, build_simple_adadelta)

            if checkpoint.get('optimizer_state_dict', None) is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except ValueError as e:
                    raise ValueError("Failed to load optimizer from %s" % filename) from e
            else:
                tlogger.info("Attempted to load optimizer to resume training, but optimizer not saved.  Creating new optimizer")

            scheduler = build_scheduler(model.args, optimizer, first_optimizer=build_simple_adadelta)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            optimizer = None
            scheduler = None

        logger.debug("-- MODEL CONFIG --")
        for k in model.args.keys():
            logger.debug("  --%s: %s", k, model.args[k])

        return Trainer(model=model, optimizer=optimizer, scheduler=scheduler, epochs_trained=epochs_trained, batches_trained=batches_trained, best_f1=best_f1, best_epoch=best_epoch)

    @staticmethod
    def build_trainer(args, train_transitions, train_constituents, tags, words, rare_words, root_labels, open_nodes, unary_limit, foundation_cache, model_load_file):
        # TODO: turn finetune, relearn_structure, multistage into an enum?
        # finetune just means continue learning, so checkpoint is sufficient
        # relearn_structure is essentially a one stage multistage
        # multistage with a checkpoint will have the proper optimizer for that epoch
        # and no special learning mode means we are training a new model and should continue
        if args['checkpoint'] and args['checkpoint_save_name'] and os.path.exists(args['checkpoint_save_name']):
            tlogger.info("Found checkpoint to continue training: %s", args['checkpoint_save_name'])
            trainer = Trainer.load(args['checkpoint_save_name'], args, load_optimizer=True, foundation_cache=foundation_cache)
            return trainer

        # in the 'finetune' case, this will preload the models into foundation_cache,
        # so the effort is not wasted
        pt = foundation_cache.load_pretrain(args['wordvec_pretrain_file'])
        forward_charlm = foundation_cache.load_charlm(args['charlm_forward_file'])
        backward_charlm = foundation_cache.load_charlm(args['charlm_backward_file'])

        if args['finetune']:
            tlogger.info("Loading model to finetune: %s", model_load_file)
            trainer = Trainer.load(model_load_file, args, load_optimizer=True, foundation_cache=NoTransformerFoundationCache(foundation_cache))
            # a new finetuning will start with a new epochs_trained count
            trainer.epochs_trained = 0
            return trainer

        if args['relearn_structure']:
            tlogger.info("Loading model to continue training with new structure from %s", model_load_file)
            temp_args = dict(args)
            # remove the pattn & lattn layers unless the saved model had them
            temp_args.pop('pattn_num_layers', None)
            temp_args.pop('lattn_d_proj', None)
            trainer = Trainer.load(model_load_file, temp_args, load_optimizer=False, foundation_cache=NoTransformerFoundationCache(foundation_cache))

            # using the model's current values works for if the new
            # dataset is the same or smaller
            # TODO: handle a larger dataset as well
            model = LSTMModel(pt,
                              forward_charlm,
                              backward_charlm,
                              trainer.model.bert_model,
                              trainer.model.bert_tokenizer,
                              trainer.model.force_bert_saved,
                              trainer.model.peft_name,
                              trainer.model.transitions,
                              trainer.model.constituents,
                              trainer.model.tags,
                              trainer.model.delta_words,
                              trainer.model.rare_words,
                              trainer.model.root_labels,
                              trainer.model.constituent_opens,
                              trainer.model.unary_limit(),
                              args)
            model = model.to(args['device'])
            model.copy_with_new_structure(trainer.model)
            optimizer = build_optimizer(args, model, False)
            scheduler = build_scheduler(args, optimizer)
            trainer = Trainer(model, optimizer, scheduler)
            return trainer

        if args['multistage']:
            # run adadelta over the model for half the time with no pattn or lattn
            # training then switches to a different optimizer for the rest
            # this works surprisingly well
            tlogger.info("Warming up model for %d iterations using AdaDelta to train the embeddings", args['epochs'] // 2)
            temp_args = dict(args)
            # remove the attention layers for the temporary model
            temp_args['pattn_num_layers'] = 0
            temp_args['lattn_d_proj'] = 0
            args = temp_args

        peft_name = None
        if args['use_peft']:
            peft_name = "constituency"
            bert_model, bert_tokenizer = load_bert(args['bert_model'])
            bert_model = build_peft_wrapper(bert_model, temp_args, tlogger, adapter_name=peft_name)
        elif args['bert_finetune'] or args['stage1_bert_finetune']:
            bert_model, bert_tokenizer = load_bert(args['bert_model'])
        else:
            bert_model, bert_tokenizer = load_bert(args['bert_model'], foundation_cache)
        model = LSTMModel(pt,
                          forward_charlm,
                          backward_charlm,
                          bert_model,
                          bert_tokenizer,
                          False,
                          peft_name,
                          train_transitions,
                          train_constituents,
                          tags,
                          words,
                          rare_words,
                          root_labels,
                          open_nodes,
                          unary_limit,
                          args)
        model = model.to(args['device'])

        optimizer = build_optimizer(args, model, build_simple_adadelta=args['multistage'])
        scheduler = build_scheduler(args, optimizer, first_optimizer=args['multistage'])

        trainer = Trainer(model, optimizer, scheduler)
        return trainer
