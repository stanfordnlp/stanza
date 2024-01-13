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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, save_name, args):
        """
        Save the model to the given path, possibly with some args

        TODO: keep all the relevant args in the model
        """
        save_dir = os.path.split(save_name)[0]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_dict = self.get_save_dict(args)
        torch.save(save_dict, save_name)
        return save_dict

    @abstractmethod
    def model_type(self):
        """
        return a ModelType
        """

    @staticmethod
    def load(filename, args=None):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from %s", filename)
            raise

        logger.debug("Loading LemmaClassifier model from %s", filename)

        model_type = checkpoint['model_type']
        if model_type is ModelType.LSTM:
            # TODO: if anyone can suggest a way to avoid this circular import
            # (or better yet, avoid the load method knowing about subclasses)
            # please do so
            # maybe the subclassing is not necessary and we just put
            # save & load in the trainer
            from stanza.models.lemma_classifier.model import LemmaClassifierLSTM

            saved_args = checkpoint['args']
            # other model args are part of the model and cannot be changed for evaluation or pipeline
            # the file paths might be relevant, though
            keep_args = ['wordvec_pretrain_file', 'charlm_forward_file', 'charlm_backward_file']
            for arg in keep_args:
                if args.get(arg, None) is not None:
                    saved_args[arg] = args[arg]

            # TODO: refactor loading the pretrain (also done in the trainer)
            pt = load_pretrain(args['wordvec_pretrain_file'])
            emb_matrix = pt.emb
            word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix))
            vocab_map = { word.replace('\xa0', ' '): i for i, word in enumerate(pt.vocab) }
            vocab_size = emb_matrix.shape[0]
            embedding_dim = emb_matrix.shape[1]

            if saved_args['use_charlm']:
                # Evaluate charlm
                model = LemmaClassifierLSTM(vocab_size=vocab_size,
                                            embedding_dim=embedding_dim,
                                            hidden_dim=saved_args['hidden_dim'],
                                            output_dim=len(checkpoint['label_decoder']),
                                            vocab_map=vocab_map,
                                            pt_embedding=word_embeddings,
                                            label_decoder=checkpoint['label_decoder'],
                                            upos_emb_dim=saved_args['upos_emb_dim'],
                                            upos_to_id=checkpoint['upos_to_id'],
                                            charlm=True,
                                            charlm_forward_file=saved_args['charlm_forward_file'],
                                            charlm_backward_file=saved_args['charlm_backward_file'])
            else:
                # Evaluate standard model (bi-LSTM with GloVe embeddings, no charlm)
                model = LemmaClassifierLSTM(vocab_size=vocab_size,
                                            embedding_dim=embedding_dim,
                                            hidden_dim=saved_args['hidden_dim'],
                                            output_dim=len(checkpoint['label_decoder']),
                                            vocab_map=vocab_map,
                                            pt_embedding=word_embeddings,
                                            label_decoder=checkpoint['label_decoder'])
        elif model_type is ModelType.TRANSFORMER:
            from stanza.models.lemma_classifier.transformer_baseline.model import LemmaClassifierWithTransformer

            output_dim = len(checkpoint['label_decoder'])
            saved_args = checkpoint['args']
            bert_model = saved_args['bert_model']
            model = LemmaClassifierWithTransformer(output_dim=output_dim, transformer_name=bert_model, label_decoder=checkpoint['label_decoder'])
        else:
            raise ValueError("Unknown model type %s" % model_type)

        model.load_state_dict(checkpoint['params'])
        return model
