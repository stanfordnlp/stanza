"""
Organizes the model itself and its optimizer in one place

Saving the optimizer allows for easy restarting of training
"""

import logging
import os
import torch
import torch.optim as optim

import stanza.models.classifiers.data as data
import stanza.models.classifiers.cnn_classifier as cnn_classifier
from stanza.models.common.foundation_cache import load_bert, load_charlm, load_pretrain
from stanza.models.common.pretrain import Pretrain

logger = logging.getLogger('stanza')

class Trainer:
    """
    Stores a constituency model and its optimizer
    """

    def __init__(self, model, optimizer=None, epochs_trained=0, global_step=0, best_score=None):
        self.model = model
        self.optimizer = optimizer
        # we keep track of position in the learning so that we can
        # checkpoint & restart if needed without restarting the epoch count
        self.epochs_trained = epochs_trained
        self.global_step = global_step
        # save the best dev score so that when reloading a checkpoint
        # of a model, we know how far we got
        self.best_score = best_score

    def save(self, filename, epochs_trained=None, skip_modules=True, save_optimizer=True):
        """
        save the current model, optimizer, and other state to filename

        epochs_trained can be passed as a parameter to handle saving at the end of an epoch
        """
        if epochs_trained is None:
            epochs_trained = self.epochs_trained
        save_dir = os.path.split(filename)[0]
        os.makedirs(save_dir, exist_ok=True)
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
            'model':          model_state,
            'config':         self.model.config,
            'labels':         self.model.labels,
            'extra_vocab':    self.model.extra_vocab,
            'epochs_trained': epochs_trained,
            'global_step':    self.global_step,
            'best_score':     self.best_score,
        }
        if save_optimizer and self.optimizer is not None:
            params['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(params, filename, _use_new_zipfile_serialization=False)
        logger.info("Model saved to {}".format(filename))

    @staticmethod
    def load(filename, args, foundation_cache=None, load_optimizer=False):
        if not os.path.exists(filename):
            if os.path.exists(os.path.join(args.save_dir, filename)):
                filename = os.path.join(args.save_dir, filename)
            else:
                raise FileNotFoundError("Cannot find model in {} or in {}".format(filename, os.path.join(args.save_dir, filename)))
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from {}".format(filename))
            raise
        logger.debug("Loaded model {}".format(filename))

        # TODO: should not be needed when all models have this value set
        setattr(checkpoint['config'], 'use_elmo', getattr(checkpoint['config'], 'use_elmo', False))
        setattr(checkpoint['config'], 'elmo_projection', getattr(checkpoint['config'], 'elmo_projection', False))
        setattr(checkpoint['config'], 'char_lowercase', getattr(checkpoint['config'], 'char_lowercase', False))
        setattr(checkpoint['config'], 'charlm_projection', getattr(checkpoint['config'], 'charlm_projection', None))
        setattr(checkpoint['config'], 'bert_model', getattr(checkpoint['config'], 'bert_model', None))
        setattr(checkpoint['config'], 'bilstm', getattr(checkpoint['config'], 'bilstm', False))
        setattr(checkpoint['config'], 'bilstm_hidden_dim', getattr(checkpoint['config'], 'bilstm_hidden_dim', 0))
        setattr(checkpoint['config'], 'maxpool_width', getattr(checkpoint['config'], 'maxpool_width', 1))

        epochs_trained = checkpoint.get('epochs_trained', 0)
        global_step = checkpoint.get('global_step', 0)
        best_score = checkpoint.get('best_score', None)

        # TODO: the getattr is not needed when all models have this baked into the config
        model_type = getattr(checkpoint['config'], 'model_type', 'CNNClassifier')

        pretrain = Trainer.load_pretrain(args, foundation_cache)
        elmo_model = utils.load_elmo(args.elmo_model) if args.use_elmo else None
        charmodel_forward = load_charlm(args.charlm_forward_file, foundation_cache)
        charmodel_backward = load_charlm(args.charlm_backward_file, foundation_cache)

        bert_model = checkpoint['config'].bert_model
        bert_model, bert_tokenizer = load_bert(bert_model, foundation_cache)
        if model_type == 'CNNClassifier':
            extra_vocab = checkpoint.get('extra_vocab', None)
            model = cnn_classifier.CNNClassifier(pretrain=pretrain,
                                                 extra_vocab=extra_vocab,
                                                 labels=checkpoint['labels'],
                                                 charmodel_forward=charmodel_forward,
                                                 charmodel_backward=charmodel_backward,
                                                 elmo_model=elmo_model,
                                                 bert_model=bert_model,
                                                 bert_tokenizer=bert_tokenizer,
                                                 args=checkpoint['config'])
        else:
            raise ValueError("Unknown model type {}".format(model_type))
        model.load_state_dict(checkpoint['model'], strict=False)

        if args.cuda:
            model.cuda()

        logger.debug("-- MODEL CONFIG --")
        for k in model.config.__dict__:
            logger.debug("  --{}: {}".format(k, model.config.__dict__[k]))

        logger.debug("-- MODEL LABELS --")
        logger.debug("  {}".format(" ".join(model.labels)))

        optimizer = None
        if load_optimizer:
            optimizer = Trainer.build_optimizer(model, args)
            if checkpoint.get('optimizer_state_dict', None) is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                logger.info("Attempted to load optimizer to resume training, but optimizer not saved.  Creating new optimizer")

        trainer = Trainer(model, optimizer, epochs_trained, global_step, best_score)

        return trainer


    def load_pretrain(args, foundation_cache):
        if args.wordvec_pretrain_file:
            pretrain_file = args.wordvec_pretrain_file
        elif args.wordvec_type:
            pretrain_file = '{}/{}.{}.pretrain.pt'.format(args.save_dir, args.shorthand, args.wordvec_type.name.lower())
        else:
            raise RuntimeError("TODO: need to get the wv type back from get_wordvec_file")

        logger.debug("Looking for pretrained vectors in {}".format(pretrain_file))
        if os.path.exists(pretrain_file):
            return load_pretrain(pretrain_file, foundation_cache)
        elif args.wordvec_raw_file:
            vec_file = args.wordvec_raw_file
            logger.debug("Pretrain not found.  Looking in {}".format(vec_file))
        else:
            vec_file = utils.get_wordvec_file(args.wordvec_dir, args.shorthand, args.wordvec_type.name.lower())
            logger.debug("Pretrain not found.  Looking in {}".format(vec_file))
        pretrain = Pretrain(pretrain_file, vec_file, args.pretrain_max_vocab)
        logger.debug("Embedding shape: %s" % str(pretrain.emb.shape))
        return pretrain


    @staticmethod
    def build_new_model(args, train_set):
        """
        Load pretrained pieces and then build a new model
        """
        if train_set is None:
            raise ValueError("Must have a train set to build a new model - needed for labels and delta word vectors")

        pretrain = Trainer.load_pretrain(args, foundation_cache=None)
        elmo_model = utils.load_elmo(args.elmo_model) if args.use_elmo else None
        charmodel_forward = load_charlm(args.charlm_forward_file)
        charmodel_backward = load_charlm(args.charlm_backward_file)

        labels = data.dataset_labels(train_set)
        extra_vocab = data.dataset_vocab(train_set)

        bert_model, bert_tokenizer = load_bert(args.bert_model)

        model = cnn_classifier.CNNClassifier(pretrain=pretrain,
                                             extra_vocab=extra_vocab,
                                             labels=labels,
                                             charmodel_forward=charmodel_forward,
                                             charmodel_backward=charmodel_backward,
                                             elmo_model=elmo_model,
                                             bert_model=bert_model,
                                             bert_tokenizer=bert_tokenizer,
                                             args=args)

        if args.cuda:
            model.cuda()

        optimizer = Trainer.build_optimizer(model, args)

        return Trainer(model, optimizer)


    @staticmethod
    def build_optimizer(model, args):
        if args.optim.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim.lower() == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optim.lower() == 'madgrad':
            try:
                import madgrad
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError("Could not create madgrad optimizer.  Perhaps the madgrad package is not installed") from e
            optimizer = madgrad.MADGRAD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError("Unknown optimizer: %s" % args.optim)
        return optimizer
