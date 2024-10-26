"""
A classifier that uses a constituency parser for the base embeddings
"""

import dataclasses
import logging
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from stanza.models.classifiers.base_classifier import BaseClassifier
from stanza.models.classifiers.config import ConstituencyConfig
from stanza.models.classifiers.data import SentimentDatum
from stanza.models.classifiers.utils import ModelType, build_output_layers

from stanza.models.common.utils import split_into_batches, sort_with_indices, unsort

logger = logging.getLogger('stanza')
tlogger = logging.getLogger('stanza.classifiers.trainer')

class ConstituencyClassifier(BaseClassifier):
    def __init__(self, tree_embedding, labels, args):
        super(ConstituencyClassifier, self).__init__()
        self.labels = labels
        # we build a separate config out of the args so that we can easily save it in torch
        self.config = ConstituencyConfig(fc_shapes = args.fc_shapes,
                                         dropout = args.dropout,
                                         num_classes = len(labels),
                                         constituency_backprop = args.constituency_backprop,
                                         constituency_batch_norm = args.constituency_batch_norm,
                                         constituency_node_attn = args.constituency_node_attn,
                                         constituency_top_layer = args.constituency_top_layer,
                                         constituency_all_words = args.constituency_all_words,
                                         model_type = ModelType.CONSTITUENCY)

        self.tree_embedding = tree_embedding

        self.fc_layers = build_output_layers(self.tree_embedding.output_size, self.config.fc_shapes, self.config.num_classes)
        self.dropout = nn.Dropout(self.config.dropout)

    def is_unsaved_module(self, name):
        return False

    def log_configuration(self):
        tlogger.info("Backprop into parser: %s", self.config.constituency_backprop)
        tlogger.info("Batch norm: %s", self.config.constituency_batch_norm)
        tlogger.info("Word positions used: %s", "all words" if self.config.constituency_all_words else "start and end words")
        tlogger.info("Attention over nodes: %s", self.config.constituency_node_attn)
        tlogger.info("Intermediate layers: %s", self.config.fc_shapes)

    def log_norms(self):
        lines = ["NORMS FOR MODEL PARAMTERS"]
        lines.extend(["tree_embedding." + x for x in self.tree_embedding.get_norms()])
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith('tree_embedding.'):
                lines.append("%s %.6g" % (name, torch.norm(param).item()))
        logger.info("\n".join(lines))


    def forward(self, inputs):
        inputs = [x.constituency if isinstance(x, SentimentDatum) else x for x in inputs]

        embedding = self.tree_embedding.embed_trees(inputs)
        previous_layer = torch.stack([torch.max(x, dim=0)[0] for x in embedding], dim=0)
        previous_layer = self.dropout(previous_layer)
        for fc in self.fc_layers[:-1]:
            # relu cause many neuron die
            previous_layer = self.dropout(F.gelu(fc(previous_layer)))
        out = self.fc_layers[-1](previous_layer)
        return out

    def get_params(self, skip_modules=True):
        model_state = self.state_dict()
        # skip all of the constituency parameters here -
        # we will add them by calling the model's get_params()
        skipped = [k for k in model_state.keys() if k.startswith("tree_embedding.")]
        for k in skipped:
            del model_state[k]

        tree_embedding = self.tree_embedding.get_params(skip_modules)

        config = dataclasses.asdict(self.config)
        config['model_type'] = config['model_type'].name

        params = {
            'model':           model_state,
            'tree_embedding':  tree_embedding,
            'config':          config,
            'labels':          self.labels,
        }
        return params

    def extract_sentences(self, doc):
        return [sentence.constituency for sentence in doc.sentences]
