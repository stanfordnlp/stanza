from enum import Enum
import logging
import os

import torch

from stanza.models.constituency.utils import build_optimizer, build_scheduler

logger = logging.getLogger('stanza')

class ModelType(Enum):
    LSTM               = 1
    ENSEMBLE           = 2

class BaseTrainer:
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
        params = self.model.get_params()
        checkpoint = {
            'params': params,
            'epochs_trained': self.epochs_trained,
            'batches_trained': self.batches_trained,
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
            'model_type': self.model_type,
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
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from %s", filename)
            raise
        logger.debug("Loaded model from %s", filename)

        params = checkpoint['params']

        if 'model_type' not in checkpoint:
            # old models will have this trait
            checkpoint['model_type'] = ModelType.LSTM
        if checkpoint['model_type'] == ModelType.LSTM:
            model = Trainer.model_from_params(params, checkpoint.get('bert_lora', None), args, foundation_cache, peft_name)
        elif checkpoint['model_type'] == ModelType.ENSEMBLE:
            model = EnsembleTrainer.model_from_params(params, checkpoint.get('bert_lora', None), args, foundation_cache, peft_name)
        else:
            raise ValueError("Unexpected model type: %s" % checkpoint['model_type'])

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
                logger.info("Attempted to load optimizer to resume training, but optimizer not saved.  Creating new optimizer")

            scheduler = build_scheduler(model.args, optimizer, first_optimizer=build_simple_adadelta)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            optimizer = None
            scheduler = None

        logger.debug("-- MODEL CONFIG --")
        for k in model.args.keys():
            logger.debug("  --%s: %s", k, model.args[k])

        if checkpoint['model_type'] == ModelType.LSTM:
            return Trainer(model=model, optimizer=optimizer, scheduler=scheduler, epochs_trained=epochs_trained, batches_trained=batches_trained, best_f1=best_f1, best_epoch=best_epoch)
        elif checkpoint['model_type'] == ModelType.ENSEMBLE:
            return EnsembleTrainer(model=model, optimizer=optimizer, scheduler=scheduler, epochs_trained=epochs_trained, batches_trained=batches_trained, best_f1=best_f1, best_epoch=best_epoch)
        else:
            raise ValueError("Unexpected model type: %s" % checkpoint['model_type'])

