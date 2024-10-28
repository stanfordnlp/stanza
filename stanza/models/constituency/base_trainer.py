from enum import Enum
import logging
import os

import torch

from pickle import UnpicklingError
import warnings

logger = logging.getLogger('stanza')

class ModelType(Enum):
    LSTM               = 1
    ENSEMBLE           = 2

class BaseTrainer:
    def __init__(self, model, optimizer=None, scheduler=None, epochs_trained=0, batches_trained=0, best_f1=0.0, best_epoch=0, first_optimizer=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # keeping track of the epochs trained will be useful
        # for adjusting the learning scheme
        self.epochs_trained = epochs_trained
        self.batches_trained = batches_trained
        self.best_f1 = best_f1
        self.best_epoch = best_epoch
        self.first_optimizer = first_optimizer

    def save(self, filename, save_optimizer=True):
        params = self.model.get_params()
        checkpoint = {
            'params': params,
            'epochs_trained': self.epochs_trained,
            'batches_trained': self.batches_trained,
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
            'model_type': self.model_type.name,
            'first_optimizer': self.first_optimizer,
        }
        checkpoint["bert_lora"] = self.get_peft_params()
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

    # TODO: make ABC with methods such as model_from_params?
    # TODO: if we save the type in the checkpoint, use that here to figure out which to load
    @staticmethod
    def load(filename, args=None, load_optimizer=False, foundation_cache=None, peft_name=None):
        """
        Load back a model and possibly its optimizer.
        """
        # hide the import here to avoid circular imports
        from stanza.models.constituency.ensemble import EnsembleTrainer
        from stanza.models.constituency.trainer import Trainer

        if not os.path.exists(filename):
            if args.get('save_dir', None) is None:
                raise FileNotFoundError("Cannot find model in {} and args['save_dir'] is None".format(filename))
            elif os.path.exists(os.path.join(args['save_dir'], filename)):
                filename = os.path.join(args['save_dir'], filename)
            else:
                raise FileNotFoundError("Cannot find model in {} or in {}".format(filename, os.path.join(args['save_dir'], filename)))
        try:
            # TODO: currently cannot switch this to weights_only=True
            # without in some way changing the model to save enums in
            # a safe manner, probably by converting to int
            try:
                checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
            except UnpicklingError as e:
                checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=False)
                warnings.warn("The saved constituency parser has an old format using Enum, set, unsanitized Transitions, etc.  This version of Stanza can support reading both the new and the old formats.  Future versions will only allow loading with weights_only=True.  Please resave the constituency parser using this version ASAP.")
        except BaseException:
            logger.exception("Cannot load model from %s", filename)
            raise
        logger.debug("Loaded model from %s", filename)

        params = checkpoint['params']

        if 'model_type' not in checkpoint:
            # old models will have this trait
            # TODO: can remove this after 1.10
            checkpoint['model_type'] = ModelType.LSTM
        if isinstance(checkpoint['model_type'], str):
            checkpoint['model_type'] = ModelType[checkpoint['model_type']]
        if checkpoint['model_type'] == ModelType.LSTM:
            clazz = Trainer
        elif checkpoint['model_type'] == ModelType.ENSEMBLE:
            clazz = EnsembleTrainer
        else:
            raise ValueError("Unexpected model type: %s" % checkpoint['model_type'])
        model = clazz.model_from_params(params, checkpoint.get('bert_lora', None), args, foundation_cache, peft_name)

        epochs_trained = checkpoint['epochs_trained']
        batches_trained = checkpoint.get('batches_trained', 0)
        best_f1 = checkpoint['best_f1']
        best_epoch = checkpoint['best_epoch']

        if 'first_optimizer' not in checkpoint:
            # this will only apply to old (LSTM) Trainers
            # EnsembleTrainers will always have this value saved
            # so here we can compensate by looking at the old training statistics...
            # we use params['config'] here instead of model.args
            # because the args might have a different training
            # mechanism, but in order to reload the optimizer, we need
            # to match the optimizer we build with the one that was
            # used at training time
            build_simple_adadelta = params['config']['multistage'] and epochs_trained < params['config']['epochs'] // 2
            checkpoint['first_optimizer'] = build_simple_adadelta
        first_optimizer = checkpoint['first_optimizer']

        if load_optimizer:
            optimizer = clazz.load_optimizer(model, checkpoint, first_optimizer, filename)
            scheduler = clazz.load_scheduler(model, optimizer, checkpoint, first_optimizer)
        else:
            optimizer = None
            scheduler = None

        if checkpoint['model_type'] == ModelType.LSTM:
            logger.debug("-- MODEL CONFIG --")
            for k in model.args.keys():
                logger.debug("  --%s: %s", k, model.args[k])
            return Trainer(model=model, optimizer=optimizer, scheduler=scheduler, epochs_trained=epochs_trained, batches_trained=batches_trained, best_f1=best_f1, best_epoch=best_epoch, first_optimizer=first_optimizer)
        elif checkpoint['model_type'] == ModelType.ENSEMBLE:
            return EnsembleTrainer(ensemble=model, optimizer=optimizer, scheduler=scheduler, epochs_trained=epochs_trained, batches_trained=batches_trained, best_f1=best_f1, best_epoch=best_epoch, first_optimizer=first_optimizer)
        else:
            raise ValueError("Unexpected model type: %s" % checkpoint['model_type'])

