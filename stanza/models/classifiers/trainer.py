"""
Organizes the model itself and its optimizer in one place

Saving the optimizer allows for easy restarting of training
"""

import logging
import os
import torch
import torch.optim as optim
from types import SimpleNamespace

import stanza.models.classifiers.data as data
import stanza.models.classifiers.cnn_classifier as cnn_classifier
import stanza.models.classifiers.constituency_classifier as constituency_classifier
from stanza.models.classifiers.config import CNNConfig, ConstituencyConfig
from stanza.models.classifiers.utils import ModelType, WVType, ExtraVectors
from stanza.models.common.foundation_cache import load_bert, load_bert_with_peft, load_charlm, load_pretrain
from stanza.models.common.peft_config import build_peft_wrapper, load_peft_wrapper
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.utils import get_split_optimizer
from stanza.models.constituency.tree_embedding import TreeEmbedding

from pickle import UnpicklingError
import warnings

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
            params['optimizer_state_dict'] = {opt_name: opt.state_dict() for opt_name, opt in self.optimizer.items()}
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
            # TODO: can remove the try/except once the new version is out
            #checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
            try:
                checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
            except UnpicklingError as e:
                checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=False)
                warnings.warn("The saved classifier has an old format using SimpleNamespace and/or Enum instead of a dict to store config.  This version of Stanza can support reading both the new and the old formats.  Future versions will only allow loading with weights_only=True.  Please resave the pretrained classifier using this version ASAP.")
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
        # TODO: this can be removed once v1.10.0 is out
        if isinstance(model_params['config'], SimpleNamespace):
            model_params['config'] = vars(model_params['config'])
        # TODO: these isinstance can go away after 1.10.0
        model_type = model_params['config']['model_type']
        if isinstance(model_type, str):
            model_type = ModelType[model_type]
            model_params['config']['model_type'] = model_type

        if model_type == ModelType.CNN:
            # TODO: these updates are only necessary during the
            # transition to the @dataclass version of the config
            # Once those are all saved, it is no longer necessary
            # to patch existing models (since they will all be patched)
            if 'has_charlm_forward' not in model_params['config']:
                model_params['config']['has_charlm_forward'] = args.charlm_forward_file is not None
            if 'has_charlm_backward' not in model_params['config']:
                model_params['config']['has_charlm_backward'] = args.charlm_backward_file is not None
            for argname in ['bert_hidden_layers', 'bert_finetune', 'force_bert_saved', 'use_peft',
                            'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_modules_to_save', 'lora_target_modules']:
                model_params['config'][argname] = model_params['config'].get(argname, None)
            # TODO: these isinstance can go away after 1.10.0
            if isinstance(model_params['config']['wordvec_type'], str):
                model_params['config']['wordvec_type'] = WVType[model_params['config']['wordvec_type']]
            if isinstance(model_params['config']['extra_wordvec_method'], str):
                model_params['config']['extra_wordvec_method'] = ExtraVectors[model_params['config']['extra_wordvec_method']]
            model_params['config'] = CNNConfig(**model_params['config'])

            pretrain = Trainer.load_pretrain(args, foundation_cache)
            elmo_model = utils.load_elmo(args.elmo_model) if args.use_elmo else None

            if model_params['config'].has_charlm_forward:
                charmodel_forward = load_charlm(args.charlm_forward_file, foundation_cache)
            else:
                charmodel_forward = None
            if model_params['config'].has_charlm_backward:
                charmodel_backward = load_charlm(args.charlm_backward_file, foundation_cache)
            else:
                charmodel_backward = None

            bert_model = model_params['config'].bert_model
            # TODO: can get rid of the getattr after rebuilding all models
            use_peft = getattr(model_params['config'], 'use_peft', False)
            force_bert_saved = getattr(model_params['config'], 'force_bert_saved', False)
            peft_name = None
            if use_peft:
                # if loading a peft model, we first load the base transformer
                # the CNNClassifier code wraps the transformer in peft
                # after creating the CNNClassifier with the peft wrapper,
                # we *then* load the weights
                bert_model, bert_tokenizer, peft_name = load_bert_with_peft(bert_model, "classifier", foundation_cache)
                bert_model = load_peft_wrapper(bert_model, model_params['bert_lora'], vars(model_params['config']), logger, peft_name)
            elif force_bert_saved:
                bert_model, bert_tokenizer = load_bert(bert_model)
            else:
                bert_model, bert_tokenizer = load_bert(bert_model, foundation_cache)
            model = cnn_classifier.CNNClassifier(pretrain=pretrain,
                                                 extra_vocab=model_params['extra_vocab'],
                                                 labels=model_params['labels'],
                                                 charmodel_forward=charmodel_forward,
                                                 charmodel_backward=charmodel_backward,
                                                 elmo_model=elmo_model,
                                                 bert_model=bert_model,
                                                 bert_tokenizer=bert_tokenizer,
                                                 force_bert_saved=force_bert_saved,
                                                 peft_name=peft_name,
                                                 args=model_params['config'])
        elif model_type == ModelType.CONSTITUENCY:
            # the constituency version doesn't have a peft feature yet
            use_peft = False
            pretrain_args = {
                'wordvec_pretrain_file': args.wordvec_pretrain_file,
                'charlm_forward_file': args.charlm_forward_file,
                'charlm_backward_file': args.charlm_backward_file,
            }
            # TODO: integrate with peft for the constituency version
            tree_embedding = TreeEmbedding.model_from_params(model_params['tree_embedding'], pretrain_args, foundation_cache)
            model_params['config'] = ConstituencyConfig(**model_params['config'])
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
                for opt_name, opt_state_dict in checkpoint['optimizer_state_dict'].items():
                    optimizer[opt_name].load_state_dict(opt_state_dict)
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
            peft_name = None
            bert_model, bert_tokenizer = load_bert(args.bert_model)

            use_peft = getattr(args, "use_peft", False)
            if use_peft:
                peft_name = "sentiment"
                bert_model = build_peft_wrapper(bert_model, vars(args), logger, adapter_name=peft_name)

            extra_vocab = data.dataset_vocab(train_set)
            force_bert_saved = args.bert_finetune
            model = cnn_classifier.CNNClassifier(pretrain=pretrain,
                                                 extra_vocab=extra_vocab,
                                                 labels=labels,
                                                 charmodel_forward=charmodel_forward,
                                                 charmodel_backward=charmodel_backward,
                                                 elmo_model=elmo_model,
                                                 bert_model=bert_model,
                                                 bert_tokenizer=bert_tokenizer,
                                                 force_bert_saved=force_bert_saved,
                                                 peft_name=peft_name,
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
        return get_split_optimizer(args.optim.lower(), model, args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, bert_learning_rate=args.bert_learning_rate, bert_weight_decay=args.weight_decay * args.bert_weight_decay, is_peft=args.use_peft)
