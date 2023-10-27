"""
Base class for the LemmaClassifier types.

Versions include LSTM and Transformer varieties
"""

import logging

from abc import ABC, abstractmethod

import os

import torch
import torch.nn as nn

from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.lemma_classifier.constants import ModelType

logger = logging.getLogger('stanza.lemmaclassifier')

class LemmaClassifier(ABC, nn.Module):
    def __init__(self, label_decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_decoder = label_decoder
        self.unsaved_modules = []

    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def is_unsaved_module(self, name):
        return name.split('.')[0] in self.unsaved_modules

    def save(self, save_name):
        """
        Save the model to the given path, possibly with some args
        """
        save_dir = os.path.split(save_name)[0]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_name)
        return save_dict

    @abstractmethod
    def model_type(self):
        """
        return a ModelType
        """

    @staticmethod
    def from_checkpoint(checkpoint, args=None):
        model_type = ModelType[checkpoint['model_type']]
        if model_type is ModelType.LSTM:
            # TODO: if anyone can suggest a way to avoid this circular import
            # (or better yet, avoid the load method knowing about subclasses)
            # please do so
            # maybe the subclassing is not necessary and we just put
            # save & load in the trainer
            from stanza.models.lemma_classifier.lstm_model import LemmaClassifierLSTM

            saved_args = checkpoint['args']
            # other model args are part of the model and cannot be changed for evaluation or pipeline
            # the file paths might be relevant, though
            keep_args = ['wordvec_pretrain_file', 'charlm_forward_file', 'charlm_backward_file']
            for arg in keep_args:
                if args is not None and args.get(arg, None) is not None:
                    saved_args[arg] = args[arg]

            # TODO: refactor loading the pretrain (also done in the trainer)
            pt = load_pretrain(saved_args['wordvec_pretrain_file'])

            use_charlm = saved_args['use_charlm']
            charlm_forward_file = saved_args.get('charlm_forward_file', None)
            charlm_backward_file = saved_args.get('charlm_backward_file', None)

            model = LemmaClassifierLSTM(model_args=saved_args,
                                        output_dim=len(checkpoint['label_decoder']),
                                        pt_embedding=pt,
                                        label_decoder=checkpoint['label_decoder'],
                                        upos_to_id=checkpoint['upos_to_id'],
                                        known_words=checkpoint['known_words'],
                                        use_charlm=use_charlm,
                                        charlm_forward_file=charlm_forward_file,
                                        charlm_backward_file=charlm_backward_file)
        elif model_type is ModelType.TRANSFORMER:
            from stanza.models.lemma_classifier.transformer_model import LemmaClassifierWithTransformer

            output_dim = len(checkpoint['label_decoder'])
            saved_args = checkpoint['args']
            bert_model = saved_args['bert_model']
            model = LemmaClassifierWithTransformer(model_args = saved_args, output_dim=output_dim, transformer_name=bert_model, label_decoder=checkpoint['label_decoder'])
        else:
            raise ValueError("Unknown model type %s" % model_type)

        # strict=False to accommodate missing parameters from the transformer or charlm
        model.load_state_dict(checkpoint['params'], strict=False)
        return model

    @staticmethod
    def load(filename, args=None):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from %s", filename)
            raise

        logger.debug("Loading LemmaClassifier model from %s", filename)

        return LemmaClassifier.from_checkpoint(checkpoint)
