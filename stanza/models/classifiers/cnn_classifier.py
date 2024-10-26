import dataclasses
import logging
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import stanza.models.classifiers.data as data
from stanza.models.classifiers.base_classifier import BaseClassifier
from stanza.models.classifiers.config import CNNConfig
from stanza.models.classifiers.data import SentimentDatum
from stanza.models.classifiers.utils import ExtraVectors, ModelType, build_output_layers
from stanza.models.common.bert_embedding import extract_bert_embeddings
from stanza.models.common.data import get_long_tensor, sort_all
from stanza.models.common.utils import attach_bert_model
from stanza.models.common.vocab import PAD_ID, UNK_ID

"""
The CNN classifier is based on Yoon Kim's work:

https://arxiv.org/abs/1408.5882

Also included are maxpool 2d, conv 2d, and a bilstm, as in

Text Classification Improved by Integrating Bidirectional LSTM
with Two-dimensional Max Pooling
https://aclanthology.org/C16-1329.pdf

The architecture is simple:

- Embedding at the bottom layer
  - separate learnable entry for UNK, since many of the embeddings we have use 0 for UNK
- maybe a bilstm layer, as per a command line flag
- Some number of conv2d layers over the embedding
- Maxpool layers over small windows, window size being a parameter
- FC layer to the classification layer

One experiment which was run and found to be a bit of a negative was
putting a layer on top of the pretrain.  You would think that might
help, but dev performance went down for each variation of
  - trans(emb)
  - relu(trans(emb))
  - dropout(trans(emb))
  - dropout(relu(trans(emb)))
"""

logger = logging.getLogger('stanza')
tlogger = logging.getLogger('stanza.classifiers.trainer')

class CNNClassifier(BaseClassifier):
    def __init__(self, pretrain, extra_vocab, labels,
                 charmodel_forward, charmodel_backward, elmo_model, bert_model, bert_tokenizer, force_bert_saved, peft_name,
                 args):
        """
        pretrain is a pretrained word embedding.  should have .emb and .vocab

        extra_vocab is a collection of words in the training data to
        be used for the delta word embedding, if used.  can be set to
        None if delta word embedding is not used.

        labels is the list of labels we expect in the training data.
        Used to derive the number of classes.  Saving it in the model
        will let us check that test data has the same labels

        args is either the complete arguments when training, or the
        subset of arguments stored in the model save file
        """
        super(CNNClassifier, self).__init__()
        self.labels = labels
        bert_finetune = args.bert_finetune
        use_peft = args.use_peft
        force_bert_saved = force_bert_saved or bert_finetune
        logger.debug("bert_finetune %s / force_bert_saved %s", bert_finetune, force_bert_saved)

        # this may change when loaded in a new Pipeline, so it's not part of the config
        self.peft_name = peft_name

        # we build a separate config out of the args so that we can easily save it in torch
        self.config = CNNConfig(filter_channels = args.filter_channels,
                                filter_sizes = args.filter_sizes,
                                fc_shapes = args.fc_shapes,
                                dropout = args.dropout,
                                num_classes = len(labels),
                                wordvec_type = args.wordvec_type,
                                extra_wordvec_method = args.extra_wordvec_method,
                                extra_wordvec_dim = args.extra_wordvec_dim,
                                extra_wordvec_max_norm = args.extra_wordvec_max_norm,
                                char_lowercase = args.char_lowercase,
                                charlm_projection = args.charlm_projection,
                                has_charlm_forward = charmodel_forward is not None,
                                has_charlm_backward = charmodel_backward is not None,
                                use_elmo = args.use_elmo,
                                elmo_projection = args.elmo_projection,
                                bert_model = args.bert_model,
                                bert_finetune = bert_finetune,
                                bert_hidden_layers = args.bert_hidden_layers,
                                force_bert_saved = force_bert_saved,

                                use_peft = use_peft,
                                lora_rank = args.lora_rank,
                                lora_alpha = args.lora_alpha,
                                lora_dropout = args.lora_dropout,
                                lora_modules_to_save = args.lora_modules_to_save,
                                lora_target_modules = args.lora_target_modules,

                                bilstm = args.bilstm,
                                bilstm_hidden_dim = args.bilstm_hidden_dim,
                                maxpool_width = args.maxpool_width,
                                model_type = ModelType.CNN)

        self.char_lowercase = args.char_lowercase

        self.unsaved_modules = []

        emb_matrix = pretrain.emb
        self.add_unsaved_module('embedding', nn.Embedding.from_pretrained(emb_matrix, freeze=True))
        self.add_unsaved_module('elmo_model', elmo_model)
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        self.add_unsaved_module('forward_charlm', charmodel_forward)
        if charmodel_forward is not None:
            tlogger.debug("Got forward char model of dimension {}".format(charmodel_forward.hidden_dim()))
            if not charmodel_forward.is_forward_lm:
                raise ValueError("Got a backward charlm as a forward charlm!")
        self.add_unsaved_module('backward_charlm', charmodel_backward)
        if charmodel_backward is not None:
            tlogger.debug("Got backward char model of dimension {}".format(charmodel_backward.hidden_dim()))
            if charmodel_backward.is_forward_lm:
                raise ValueError("Got a forward charlm as a backward charlm!")

        attach_bert_model(self, bert_model, bert_tokenizer, self.config.use_peft, force_bert_saved)

        # The Pretrain has PAD and UNK already (indices 0 and 1), but we
        # possibly want to train UNK while freezing the rest of the embedding
        # note that the /10.0 operation has to be inside nn.Parameter unless
        # you want to spend a long time debugging this
        self.unk = nn.Parameter(torch.randn(self.embedding_dim) / np.sqrt(self.embedding_dim) / 10.0)

        # replacing NBSP picks up a whole bunch of words for VI
        self.vocab_map = { word.replace('\xa0', ' '): i for i, word in enumerate(pretrain.vocab) }

        if self.config.extra_wordvec_method is not ExtraVectors.NONE:
            if not extra_vocab:
                raise ValueError("Should have had extra_vocab set for extra_wordvec_method {}".format(self.config.extra_wordvec_method))
            if not args.extra_wordvec_dim:
                self.config.extra_wordvec_dim = self.embedding_dim
            if self.config.extra_wordvec_method is ExtraVectors.SUM:
                if self.config.extra_wordvec_dim != self.embedding_dim:
                    raise ValueError("extra_wordvec_dim must equal embedding_dim for {}".format(self.config.extra_wordvec_method))

            self.extra_vocab = list(extra_vocab)
            self.extra_vocab_map = { word: i for i, word in enumerate(self.extra_vocab) }
            # TODO: possibly add regularization specifically on the extra embedding?
            # note: it looks like a bug that this doesn't add UNK or PAD, but actually
            # those are expected to already be the first two entries
            self.extra_embedding = nn.Embedding(num_embeddings = len(extra_vocab),
                                                embedding_dim = self.config.extra_wordvec_dim,
                                                max_norm = self.config.extra_wordvec_max_norm,
                                                padding_idx = 0)
            tlogger.debug("Extra embedding size: {}".format(self.extra_embedding.weight.shape))
        else:
            self.extra_vocab = None
            self.extra_vocab_map = None
            self.config.extra_wordvec_dim = 0
            self.extra_embedding = None

        # Pytorch is "aware" of the existence of the nn.Modules inside
        # an nn.ModuleList in terms of parameters() etc
        if self.config.extra_wordvec_method is ExtraVectors.NONE:
            total_embedding_dim = self.embedding_dim
        elif self.config.extra_wordvec_method is ExtraVectors.SUM:
            total_embedding_dim = self.embedding_dim
        elif self.config.extra_wordvec_method is ExtraVectors.CONCAT:
            total_embedding_dim = self.embedding_dim + self.config.extra_wordvec_dim
        else:
            raise ValueError("unable to handle {}".format(self.config.extra_wordvec_method))

        if charmodel_forward is not None:
            if args.charlm_projection:
                self.charmodel_forward_projection = nn.Linear(charmodel_forward.hidden_dim(), args.charlm_projection)
                total_embedding_dim += args.charlm_projection
            else:
                self.charmodel_forward_projection = None
                total_embedding_dim += charmodel_forward.hidden_dim()

        if charmodel_backward is not None:
            if args.charlm_projection:
                self.charmodel_backward_projection = nn.Linear(charmodel_backward.hidden_dim(), args.charlm_projection)
                total_embedding_dim += args.charlm_projection
            else:
                self.charmodel_backward_projection = None
                total_embedding_dim += charmodel_backward.hidden_dim()

        if self.config.use_elmo:
            if elmo_model is None:
                raise ValueError("Model requires elmo, but elmo_model not passed in")
            elmo_dim = elmo_model.sents2elmo([["Test"]])[0].shape[1]

            # this mapping will combine 3 layers of elmo to 1 layer of features
            self.elmo_combine_layers = nn.Linear(in_features=3, out_features=1, bias=False)
            if self.config.elmo_projection:
                self.elmo_projection = nn.Linear(in_features=elmo_dim, out_features=self.config.elmo_projection)
                total_embedding_dim = total_embedding_dim + self.config.elmo_projection
            else:
                total_embedding_dim = total_embedding_dim + elmo_dim

        if bert_model is not None:
            if self.config.bert_hidden_layers:
                # The average will be offset by 1/N so that the default zeros
                # repressents an average of the N layers
                if self.config.bert_hidden_layers > bert_model.config.num_hidden_layers:
                    # limit ourselves to the number of layers actually available
                    # note that we can +1 because of the initial embedding layer
                    self.config.bert_hidden_layers = bert_model.config.num_hidden_layers + 1
                self.bert_layer_mix = nn.Linear(self.config.bert_hidden_layers, 1, bias=False)
                nn.init.zeros_(self.bert_layer_mix.weight)
            else:
                # an average of layers 2, 3, 4 will be used
                # (for historic reasons)
                self.bert_layer_mix = None

            if bert_tokenizer is None:
                raise ValueError("Cannot have a bert model without a tokenizer")
            self.bert_dim = self.bert_model.config.hidden_size
            total_embedding_dim += self.bert_dim

        if self.config.bilstm:
            conv_input_dim = self.config.bilstm_hidden_dim * 2
            self.bilstm = nn.LSTM(batch_first=True,
                                  input_size=total_embedding_dim,
                                  hidden_size=self.config.bilstm_hidden_dim,
                                  num_layers=2,
                                  bidirectional=True,
                                  dropout=0.2)
        else:
            conv_input_dim = total_embedding_dim
            self.bilstm = None

        self.fc_input_size = 0
        self.conv_layers = nn.ModuleList()
        self.max_window = 0
        for filter_idx, filter_size in enumerate(self.config.filter_sizes):
            if isinstance(filter_size, int):
                self.max_window = max(self.max_window, filter_size)
                if isinstance(self.config.filter_channels, int):
                    filter_channels = self.config.filter_channels
                else:
                    filter_channels = self.config.filter_channels[filter_idx]
                fc_delta = filter_channels // self.config.maxpool_width
                tlogger.debug("Adding full width filter %d.  Output channels: %d -> %d", filter_size, filter_channels, fc_delta)
                self.fc_input_size += fc_delta
                self.conv_layers.append(nn.Conv2d(in_channels=1,
                                                  out_channels=filter_channels,
                                                  kernel_size=(filter_size, conv_input_dim)))
            elif isinstance(filter_size, tuple) and len(filter_size) == 2:
                filter_height, filter_width = filter_size
                self.max_window = max(self.max_window, filter_width)
                if isinstance(self.config.filter_channels, int):
                    filter_channels = max(1, self.config.filter_channels // (conv_input_dim // filter_width))
                else:
                    filter_channels = self.config.filter_channels[filter_idx]
                fc_delta = filter_channels * (conv_input_dim // filter_width) // self.config.maxpool_width
                tlogger.debug("Adding filter %s.  Output channels: %d -> %d", filter_size, filter_channels, fc_delta)
                self.fc_input_size += fc_delta
                self.conv_layers.append(nn.Conv2d(in_channels=1,
                                                  out_channels=filter_channels,
                                                  stride=(1, filter_width),
                                                  kernel_size=(filter_height, filter_width)))
            else:
                raise ValueError("Expected int or 2d tuple for conv size")

        tlogger.debug("Input dim to FC layers: %d", self.fc_input_size)
        self.fc_layers = build_output_layers(self.fc_input_size, self.config.fc_shapes, self.config.num_classes)

        self.dropout = nn.Dropout(self.config.dropout)

    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)

        if module is not None and (name in ('forward_charlm', 'backward_charlm') or
                                   (name == 'bert_model' and not self.config.use_peft)):
            # if we are using peft, we should not save the transformer directly
            # instead, the peft parameters only will be saved later
            for _, parameter in module.named_parameters():
                parameter.requires_grad = False

    def is_unsaved_module(self, name):
        return name.split('.')[0] in self.unsaved_modules

    def log_configuration(self):
        """
        Log some essential information about the model configuration to the training logger
        """
        tlogger.info("Filter sizes: %s" % str(self.config.filter_sizes))
        tlogger.info("Filter channels: %s" % str(self.config.filter_channels))
        tlogger.info("Intermediate layers: %s" % str(self.config.fc_shapes))

    def log_norms(self):
        lines = ["NORMS FOR MODEL PARAMTERS"]
        for name, param in self.named_parameters():
            if param.requires_grad and name.split(".")[0] not in ('forward_charlm', 'backward_charlm'):
                lines.append("%s %.6g" % (name, torch.norm(param).item()))
        logger.info("\n".join(lines))

    def build_char_reps(self, inputs, max_phrase_len, charlm, projection, begin_paddings, device):
        char_reps = charlm.build_char_representation(inputs)
        if projection is not None:
            char_reps = [projection(x) for x in char_reps]
        char_inputs = torch.zeros((len(inputs), max_phrase_len, char_reps[0].shape[-1]), device=device)
        for idx, rep in enumerate(char_reps):
            start = begin_paddings[idx]
            end = start + rep.shape[0]
            char_inputs[idx, start:end, :] = rep
        return char_inputs

    def extract_bert_embeddings(self, inputs, max_phrase_len, begin_paddings, device):
        bert_embeddings = extract_bert_embeddings(self.config.bert_model, self.bert_tokenizer, self.bert_model, inputs, device,
                                                  keep_endpoints=False,
                                                  num_layers=self.bert_layer_mix.in_features if self.bert_layer_mix is not None else None,
                                                  detach=not self.config.bert_finetune,
                                                  peft_name=self.peft_name)
        if self.bert_layer_mix is not None:
            # add the average so that the default behavior is to
            # take an average of the N layers, and anything else
            # other than that needs to be learned
            bert_embeddings = [self.bert_layer_mix(feature).squeeze(2) + feature.sum(axis=2) / self.bert_layer_mix.in_features for feature in bert_embeddings]
        bert_inputs = torch.zeros((len(inputs), max_phrase_len, bert_embeddings[0].shape[-1]), device=device)
        for idx, rep in enumerate(bert_embeddings):
            start = begin_paddings[idx]
            end = start + rep.shape[0]
            bert_inputs[idx, start:end, :] = rep
        return bert_inputs

    def forward(self, inputs):
        # assume all pieces are on the same device
        device = next(self.parameters()).device

        vocab_map = self.vocab_map
        def map_word(word):
            idx = vocab_map.get(word, None)
            if idx is not None:
                return idx
            if word[-1] == "'":
                idx = vocab_map.get(word[:-1], None)
                if idx is not None:
                    return idx
            return vocab_map.get(word.lower(), UNK_ID)

        inputs = [x.text if isinstance(x, SentimentDatum) else x for x in inputs]
        # we will pad each phrase so either it matches the longest
        # conv or the longest phrase in the input, whichever is longer
        max_phrase_len = max(len(x) for x in inputs)
        if self.max_window > max_phrase_len:
            max_phrase_len = self.max_window

        batch_indices = []
        batch_unknowns = []
        extra_batch_indices = []
        begin_paddings = []
        end_paddings = []

        elmo_batch_words = []

        for phrase in inputs:
            # we use random at training time to try to learn different
            # positions of padding.  at test time, though, we want to
            # have consistent results, so we set that to 0 begin_pad
            if self.training:
                begin_pad_width = random.randint(0, max_phrase_len - len(phrase))
            else:
                begin_pad_width = 0
            end_pad_width = max_phrase_len - begin_pad_width - len(phrase)

            begin_paddings.append(begin_pad_width)
            end_paddings.append(end_pad_width)

            # the initial lists are the length of the begin padding
            sentence_indices = [PAD_ID] * begin_pad_width
            sentence_indices.extend([map_word(x) for x in phrase])
            sentence_indices.extend([PAD_ID] * end_pad_width)

            # the "unknowns" will be the locations of the unknown words.
            # these locations will get the specially trained unknown vector
            # TODO: split UNK based on part of speech?  might be an interesting experiment
            sentence_unknowns = [idx for idx, word in enumerate(sentence_indices) if word == UNK_ID]

            batch_indices.append(sentence_indices)
            batch_unknowns.append(sentence_unknowns)

            if self.extra_vocab:
                extra_sentence_indices = [PAD_ID] * begin_pad_width
                for word in phrase:
                    if word in self.extra_vocab_map:
                        # the extra vocab is initialized from the
                        # words in the training set, which means there
                        # would be no unknown words.  to occasionally
                        # train the extra vocab's unknown words, we
                        # replace 1% of the words with UNK
                        # we don't do that for the original embedding
                        # on the assumption that there may be some
                        # unknown words in the training set anyway
                        # TODO: maybe train unk for the original embedding?
                        if self.training and random.random() < 0.01:
                            extra_sentence_indices.append(UNK_ID)
                        else:
                            extra_sentence_indices.append(self.extra_vocab_map[word])
                    else:
                        extra_sentence_indices.append(UNK_ID)
                extra_sentence_indices.extend([PAD_ID] * end_pad_width)
                extra_batch_indices.append(extra_sentence_indices)

            if self.config.use_elmo:
                elmo_phrase_words = [""] * begin_pad_width
                for word in phrase:
                    elmo_phrase_words.append(word)
                elmo_phrase_words.extend([""] * end_pad_width)
                elmo_batch_words.append(elmo_phrase_words)

        # creating a single large list with all the indices lets us
        # create a single tensor, which is much faster than creating
        # many tiny tensors
        # we can convert this to the input to the CNN
        # it is padded at one or both ends so that it is now num_phrases x max_len x emb_size
        # there are two ways in which this padding is suboptimal
        # the first is that for short sentences, smaller windows will
        #   be padded to the point that some windows are entirely pad
        # the second is that a sentence S will have more or less padding
        #   depending on what other sentences are in its batch
        # we assume these effects are pretty minimal
        batch_indices = torch.tensor(batch_indices, requires_grad=False, device=device)
        input_vectors = self.embedding(batch_indices)
        # we use the random unk so that we are not necessarily
        # learning to match 0s for unk
        for phrase_num, sentence_unknowns in enumerate(batch_unknowns):
            input_vectors[phrase_num][sentence_unknowns] = self.unk

        if self.extra_vocab:
            extra_batch_indices = torch.tensor(extra_batch_indices, requires_grad=False, device=device)
            extra_input_vectors = self.extra_embedding(extra_batch_indices)
            if self.config.extra_wordvec_method is ExtraVectors.CONCAT:
                all_inputs = [input_vectors, extra_input_vectors]
            elif self.config.extra_wordvec_method is ExtraVectors.SUM:
                all_inputs = [input_vectors + extra_input_vectors]
            else:
                raise ValueError("unable to handle {}".format(self.config.extra_wordvec_method))
        else:
            all_inputs = [input_vectors]

        if self.forward_charlm is not None:
            char_reps_forward = self.build_char_reps(inputs, max_phrase_len, self.forward_charlm, self.charmodel_forward_projection, begin_paddings, device)
            all_inputs.append(char_reps_forward)

        if self.backward_charlm is not None:
            char_reps_backward = self.build_char_reps(inputs, max_phrase_len, self.backward_charlm, self.charmodel_backward_projection, begin_paddings, device)
            all_inputs.append(char_reps_backward)

        if self.config.use_elmo:
            # this will be N arrays of 3xMx1024 where M is the number of words
            # and N is the number of sentences (and 1024 is actually the number of weights)
            elmo_arrays = self.elmo_model.sents2elmo(elmo_batch_words, output_layer=-2)
            elmo_tensors = [torch.tensor(x).to(device=device) for x in elmo_arrays]
            # elmo_tensor will now be Nx3xMx1024
            elmo_tensor = torch.stack(elmo_tensors)
            # Nx1024xMx3
            elmo_tensor = torch.transpose(elmo_tensor, 1, 3)
            # NxMx1024x3
            elmo_tensor = torch.transpose(elmo_tensor, 1, 2)
            # NxMx1024x1
            elmo_tensor = self.elmo_combine_layers(elmo_tensor)
            # NxMx1024
            elmo_tensor = elmo_tensor.squeeze(3)
            if self.config.elmo_projection:
                elmo_tensor = self.elmo_projection(elmo_tensor)
            all_inputs.append(elmo_tensor)

        if self.bert_model is not None:
            bert_embeddings = self.extract_bert_embeddings(inputs, max_phrase_len, begin_paddings, device)
            all_inputs.append(bert_embeddings)

        # still works even if there's just one item
        input_vectors = torch.cat(all_inputs, dim=2)

        if self.config.bilstm:
            input_vectors, _ = self.bilstm(self.dropout(input_vectors))

        # reshape to fit the input tensors
        x = input_vectors.unsqueeze(1)

        conv_outs = []
        for conv, filter_size in zip(self.conv_layers, self.config.filter_sizes):
            if isinstance(filter_size, int):
                conv_out = self.dropout(F.relu(conv(x).squeeze(3)))
                conv_outs.append(conv_out)
            else:
                conv_out = conv(x).transpose(2, 3).flatten(1, 2)
                conv_out = self.dropout(F.relu(conv_out))
                conv_outs.append(conv_out)
        pool_outs = [F.max_pool2d(out, (self.config.maxpool_width, out.shape[2])).squeeze(2) for out in conv_outs]
        pooled = torch.cat(pool_outs, dim=1)

        previous_layer = pooled
        for fc in self.fc_layers[:-1]:
            previous_layer = self.dropout(F.relu(fc(previous_layer)))
        out = self.fc_layers[-1](previous_layer)
        # note that we return the raw logits rather than use a softmax
        # https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920/4
        return out

    def get_params(self, skip_modules=True):
        model_state = self.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if self.is_unsaved_module(k)]
            for k in skipped:
                del model_state[k]

        config = dataclasses.asdict(self.config)
        config['wordvec_type'] = config['wordvec_type'].name
        config['extra_wordvec_method'] = config['extra_wordvec_method'].name
        config['model_type'] = config['model_type'].name

        params = {
            'model':        model_state,
            'config':       config,
            'labels':       self.labels,
            'extra_vocab':  self.extra_vocab,
        }
        if self.config.use_peft:
            # Hide import so that peft dependency is optional
            from peft import get_peft_model_state_dict
            params["bert_lora"] = get_peft_model_state_dict(self.bert_model, adapter_name=self.peft_name)
        return params

    def preprocess_data(self, sentences):
        sentences = [data.update_text(s, self.config.wordvec_type) for s in sentences]
        return sentences

    def extract_sentences(self, doc):
        # TODO: tokens or words better here?
        return [[token.text for token in sentence.tokens] for sentence in doc.sentences]
