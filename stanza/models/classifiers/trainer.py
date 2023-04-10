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
import stanza.models.classifiers.constituency_classifier as constituency_classifier
from stanza.models.classifiers.utils import ModelType
from stanza.models.common.foundation_cache import load_bert, load_charlm, load_pretrain
from stanza.models.common.pretrain import Pretrain
from stanza.models.constituency.tree_embedding import TreeEmbedding

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
        model_params = self.model.get_params(skip_modules)
        params = {
            'params':         model_params,
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
            if args.save_dir is None:
                raise FileNotFoundError("Cannot find model in {} and args.save_dir is None".format(filename))
            elif os.path.exists(os.path.join(args.save_dir, filename)):
                filename = os.path.join(args.save_dir, filename)
            else:
                raise FileNotFoundError("Cannot find model in {} or in {}".format(filename, os.path.join(args.save_dir, filename)))
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from {}".format(filename))
            raise
        logger.debug("Loaded model {}".format(filename))

        epochs_trained = checkpoint.get('epochs_trained', 0)
        global_step = checkpoint.get('global_step', 0)
        best_score = checkpoint.get('best_score', None)

        # TODO: can remove this block once all models are retrained
        if 'params' not in checkpoint:
            model_params = {
                'model':        checkpoint['model'],
                'config':       checkpoint['config'],
                'labels':       checkpoint['labels'],
                'extra_vocab':  checkpoint['extra_vocab'],
            }
        else:
            model_params = checkpoint['params']
        model_type = model_params['config'].model_type

        if model_type == ModelType.CNN:
            pretrain = Trainer.load_pretrain(args, foundation_cache)
            elmo_model = utils.load_elmo(args.elmo_model) if args.use_elmo else None
            charmodel_forward = load_charlm(args.charlm_forward_file, foundation_cache)
            charmodel_backward = load_charlm(args.charlm_backward_file, foundation_cache)

            bert_model = model_params['config'].bert_model
            bert_model, bert_tokenizer = load_bert(bert_model, foundation_cache)
            model = cnn_classifier.CNNClassifier(pretrain=pretrain,
                                                 extra_vocab=model_params['extra_vocab'],
                                                 labels=model_params['labels'],
                                                 charmodel_forward=charmodel_forward,
                                                 charmodel_backward=charmodel_backward,
                                                 elmo_model=elmo_model,
                                                 bert_model=bert_model,
                                                 bert_tokenizer=bert_tokenizer,
                                                 args=model_params['config'])
        elif model_type == ModelType.CONSTITUENCY:
            pretrain_args = {
                'wordvec_pretrain_file': args.wordvec_pretrain_file,
                'charlm_forward_file': args.charlm_forward_file,
                'charlm_backward_file': args.charlm_backward_file,
            }
            tree_embedding = TreeEmbedding.model_from_params(model_params['tree_embedding'], pretrain_args, foundation_cache)
            model = constituency_classifier.ConstituencyClassifier(tree_embedding=tree_embedding,
                                                                   labels=model_params['labels'],
                                                                   args=model_params['config'])
        else:
            raise ValueError("Unknown model type {}".format(model_type))
        model.load_state_dict(model_params['model'], strict=False)
        model = model.to(args.device)

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

        labels = data.dataset_labels(train_set)

        if args.model_type == ModelType.CNN:
            pretrain = Trainer.load_pretrain(args, foundation_cache=None)
            elmo_model = utils.load_elmo(args.elmo_model) if args.use_elmo else None
            charmodel_forward = load_charlm(args.charlm_forward_file)
            charmodel_backward = load_charlm(args.charlm_backward_file)
            bert_model, bert_tokenizer = load_bert(args.bert_model)

            extra_vocab = data.dataset_vocab(train_set)
            model = cnn_classifier.CNNClassifier(pretrain=pretrain,
                                                 extra_vocab=extra_vocab,
                                                 labels=labels,
                                                 charmodel_forward=charmodel_forward,
                                                 charmodel_backward=charmodel_backward,
                                                 elmo_model=elmo_model,
                                                 bert_model=bert_model,
                                                 bert_tokenizer=bert_tokenizer,
                                                 args=args)
            model = model.to(args.device)
        elif args.model_type == ModelType.CONSTITUENCY:
            # this passes flags such as "constituency_backprop" from
            # the classifier to the TreeEmbedding as the "backprop" flag
            parser_args = { x[len("constituency_"):]: y for x, y in vars(args).items() if x.startswith("constituency_") }
            parser_args.update({
                "wordvec_pretrain_file": args.wordvec_pretrain_file,
                "charlm_forward_file": args.charlm_forward_file,
                "charlm_backward_file": args.charlm_backward_file,
                "bert_model": args.bert_model,
                # we found that finetuning from the classifier output
                # all the way to the bert layers caused the bert model
                # to go astray
                # could make this an option... but it is much less accurate
                # with the Bert finetuning
                # noting that the constituency parser itself works better
                # after finetuning, of course
                "bert_finetune": False,
                "stage1_bert_finetune": False,
            })
            logger.info("Building constituency classifier using %s as the base model" % args.constituency_model)
            tree_embedding = TreeEmbedding.from_parser_file(parser_args)
            model = constituency_classifier.ConstituencyClassifier(tree_embedding=tree_embedding,
                                                                   labels=labels,
                                                                   args=args)
            model = model.to(args.device)
        else:
            raise ValueError("Unhandled model type {}".format(args.model_type))

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
