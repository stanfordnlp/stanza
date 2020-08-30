import logging
import random
import re
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import stanza.models.classifiers.classifier_args as classifier_args
from stanza.models.common.vocab import PAD_ID, UNK_ID

"""
The CNN classifier is based on Yoon Kim's work:

https://arxiv.org/abs/1408.5882

The architecture is simple:

- Embedding at the bottom layer
  - separate learnable entry for UNK, since many of the embeddings we have use 0 for UNK
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

class CNNClassifier(nn.Module):
    def __init__(self, pretrain, extra_vocab, labels, args):
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
        self.config = SimpleNamespace(filter_channels = args.filter_channels,
                                      filter_sizes = args.filter_sizes,
                                      fc_shapes = args.fc_shapes,
                                      dropout = args.dropout,
                                      num_classes = len(labels),
                                      wordvec_type = args.wordvec_type,
                                      extra_wordvec_method = args.extra_wordvec_method,
                                      extra_wordvec_dim = args.extra_wordvec_dim,
                                      extra_wordvec_max_norm = args.extra_wordvec_max_norm,
                                      model_type = 'CNNClassifier')

        self.unsaved_modules = []

        emb_matrix = pretrain.emb
        self.add_unsaved_module('embedding', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        # The Pretrain has PAD and UNK already (indices 0 and 1), but we
        # possibly want to train UNK while freezing the rest of the embedding
        # note that the /10.0 operation has to be inside nn.Parameter unless
        # you want to spend a long time debugging this
        self.unk = nn.Parameter(torch.randn(self.embedding_dim) / np.sqrt(self.embedding_dim) / 10.0)

        self.vocab_map = { word: i for i, word in enumerate(pretrain.vocab) }

        if self.config.extra_wordvec_method is not classifier_args.ExtraVectors.NONE:
            if not extra_vocab:
                raise ValueError("Should have had extra_vocab set for extra_wordvec_method {}".format(self.config.extra_wordvec_method))
            if not args.extra_wordvec_dim:
                self.config.extra_wordvec_dim = self.embedding_dim
            if self.config.extra_wordvec_method is classifier_args.ExtraVectors.SUM:
                if self.config.extra_wordvec_dim != self.embedding_dim:
                    raise ValueError("extra_wordvec_dim must equal embedding_dim for {}".format(self.config.extra_wordvec_method))

            self.extra_vocab = list(extra_vocab)
            self.extra_vocab_map = { word: i for i, word in enumerate(self.extra_vocab) }
            # TODO: possibly add regularization specifically on the extra embedding?
            self.extra_embedding = nn.Embedding(num_embeddings = len(extra_vocab),
                                                embedding_dim = self.config.extra_wordvec_dim,
                                                max_norm = self.config.extra_wordvec_max_norm,
                                                padding_idx = 0)
            logger.debug("Extra embedding size: {}".format(self.extra_embedding.weight.shape))
        else:
            self.extra_vocab = None
            self.extra_vocab_map = None
            self.config.extra_wordvec_dim = 0
            self.extra_embedding = None

        # Pytorch is "aware" of the existence of the nn.Modules inside
        # an nn.ModuleList in terms of parameters() etc
        if self.config.extra_wordvec_method is classifier_args.ExtraVectors.NONE:
            total_embedding_dim = self.embedding_dim
        elif self.config.extra_wordvec_method is classifier_args.ExtraVectors.SUM:
            total_embedding_dim = self.embedding_dim
        elif self.config.extra_wordvec_method is classifier_args.ExtraVectors.CONCAT:
            total_embedding_dim = self.embedding_dim + self.config.extra_wordvec_dim
        else:
            raise ValueError("unable to handle {}".format(self.config.extra_wordvec_method))

        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1,
                                                    out_channels=self.config.filter_channels,
                                                    kernel_size=(filter_size, total_embedding_dim))
                                          for filter_size in self.config.filter_sizes])

        previous_layer_size = len(self.config.filter_sizes) * self.config.filter_channels
        fc_layers = []
        for shape in self.config.fc_shapes:
            fc_layers.append(nn.Linear(previous_layer_size, shape))
            previous_layer_size = shape
        fc_layers.append(nn.Linear(previous_layer_size, self.config.num_classes))
        self.fc_layers = nn.ModuleList(fc_layers)

        self.max_window = max(self.config.filter_sizes)

        self.dropout = nn.Dropout(self.config.dropout)


    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def forward(self, inputs, device=None):
        if not device:
            # assume all pieces are on the same device
            device = next(self.parameters()).device

        # we will pad each phrase so either it matches the longest
        # conv or the longest phrase in the input, whichever is longer
        max_phrase_len = max(len(x) for x in inputs)
        if self.max_window > max_phrase_len:
            max_phrase_len = self.max_window

        batch_indices = []
        batch_unknowns = []
        extra_batch_indices = []
        for phrase in inputs:
            # TODO: random is good for train mode.  try something else at test time?
            begin_pad_width = random.randint(0, max_phrase_len - len(phrase))
            end_pad_width = max_phrase_len - begin_pad_width - len(phrase)

            # the initial lists are the length of the begin padding
            sentence_indices = [PAD_ID] * begin_pad_width
            extra_sentence_indices = [PAD_ID] * begin_pad_width
            # the "unknowns" will be the locations of the unknown words.
            # these locations will get the specially trained unknown vector
            sentence_unknowns = []

            for word in phrase:
                if word in self.vocab_map:
                    sentence_indices.append(self.vocab_map[word])
                    continue
                new_word = word.replace("-", "")
                # google vectors have words which are all dashes
                if len(new_word) == 0:
                    new_word = word
                if new_word in self.vocab_map:
                    sentence_indices.append(self.vocab_map[new_word])
                    continue

                if new_word[-1] == "'":
                    new_word = new_word[:-1]
                    if new_word in self.vocab_map:
                        sentence_indices.append(self.vocab_map[new_word])
                        continue

                # TODO: split UNK based on part of speech?  might be an interesting experiment
                sentence_unknowns.append(len(sentence_indices))
                sentence_indices.append(PAD_ID)
            if self.extra_vocab:
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

            for i in range(end_pad_width):
                sentence_indices.append(PAD_ID)
                extra_sentence_indices.append(PAD_ID)

            batch_indices.append(sentence_indices)
            batch_unknowns.append(sentence_unknowns)
            extra_batch_indices.append(extra_sentence_indices)

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
            for unknown in sentence_unknowns:
                input_vectors[phrase_num, unknown, :] = self.unk
        if self.extra_vocab:
            extra_batch_indices = torch.tensor(extra_batch_indices, requires_grad=False, device=device)
            extra_input_vectors = self.extra_embedding(extra_batch_indices)
            if self.config.extra_wordvec_method is classifier_args.ExtraVectors.CONCAT:
                input_vectors = torch.cat([input_vectors, extra_input_vectors], dim=2)
            elif self.config.extra_wordvec_method is classifier_args.ExtraVectors.SUM:
                input_vectors = input_vectors + extra_input_vectors
            else:
                raise ValueError("unable to handle {}".format(self.config.extra_wordvec_method))

        # reshape to fit the input tensors
        x = input_vectors.unsqueeze(1)

        conv_outs = [self.dropout(F.relu(conv(x).squeeze(3)))
                     for conv in self.conv_layers]
        pool_outs = [F.max_pool1d(out, out.shape[2]).squeeze(2) for out in conv_outs]
        pooled = torch.cat(pool_outs, dim=1)

        previous_layer = pooled
        for fc in self.fc_layers[:-1]:
            previous_layer = self.dropout(F.relu(fc(previous_layer)))
        out = self.fc_layers[-1](previous_layer)
        return out


# TODO: make some of the following methods part of the class

# TODO: all this code is basically the same as for POS and NER.  Should refactor
def save(filename, model, skip_modules=True):
    model_state = model.state_dict()
    # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
    if skip_modules:
        skipped = [k for k in model_state.keys() if k.split('.')[0] in model.unsaved_modules]
        for k in skipped:
            del model_state[k]
    params = {
        'model': model_state,
        'config': model.config,
        'labels': model.labels,
        'extra_vocab': model.extra_vocab,
    }
    try:
        torch.save(params, filename)
        logger.info("Model saved to {}".format(filename))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        logger.warning("Saving failed to {}... continuing anyway.  Error: {}".format(filename, e))

def load(filename, pretrain):
    try:
        checkpoint = torch.load(filename, lambda storage, loc: storage)
    except BaseException:
        logger.exception("Cannot load model from {}".format(filename))
        raise
    logger.debug("Loaded model {}".format(filename))

    model_type = getattr(checkpoint['config'], 'model_type', 'CNNClassifier')
    if model_type == 'CNNClassifier':
        extra_vocab = checkpoint.get('extra_vocab', None)
        model = CNNClassifier(pretrain,
                              extra_vocab,
                              checkpoint['labels'],
                              checkpoint['config'])
    else:
        raise ValueError("Unknown model type {}".format(model_type))
    model.load_state_dict(checkpoint['model'], strict=False)

    logger.debug("-- MODEL CONFIG --")
    for k in model.config.__dict__:
        logger.debug("  --{}: {}".format(k, model.config.__dict__[k]))

    logger.debug("-- MODEL LABELS --")
    logger.debug("  {}".format(" ".join(model.labels)))

    return model


def update_text(sentence, wordvec_type):
    """
    Process a line of text (with tokenization provided as whitespace)
    into a list of strings.
    """
    # stanford sentiment dataset has a lot of random - and /
    sentence = sentence.replace("-", " ")
    sentence = sentence.replace("/", " ")
    sentence = sentence.split()
    # our current word vectors are all entirely lowercased
    sentence = [word.lower() for word in sentence]
    if wordvec_type == classifier_args.WVType.WORD2VEC:
        return sentence
    elif wordvec_type == classifier_args.WVType.GOOGLE:
        new_sentence = []
        for word in sentence:
            if word != '0' and word != '1':
                word = re.sub('[0-9]', '#', word)
            new_sentence.append(word)
        return new_sentence
    elif wordvec_type == classifier_args.WVType.FASTTEXT:
        return sentence
    elif wordvec_type == classifier_args.WVType.OTHER:
        return sentence
    else:
        raise ValueError("Unknown wordvec_type {}".format(wordvec_type))


def label_text(model, text, batch_size=None, reverse_label_map=None, device=None):
    """
    Given a list of sentences, return the model's results on that text.
    """
    model.eval()
    if reverse_label_map is None:
        reverse_label_map = {x: y for (x, y) in enumerate(model.labels)}
    if device is None:
        device = next(model.parameters()).device

    text = [update_text(s, model.config.wordvec_type) for s in text]

    if batch_size is None:
        intervals = [(0, len(text))]
    else:
        # TODO: results would be better if we sort by length and then unsort
        intervals = [(i, min(i+batch_size, len(text))) for i in range(0, len(text), batch_size)]
    labels = []
    for interval in intervals:
        output = model(text[interval[0]:interval[1]], device)
        predicted = torch.argmax(output, dim=1)
        labels.extend(predicted.tolist())

    logger.debug("Found labels")
    for (label, sentence) in zip(labels, text):
        logger.debug((label, sentence))

    return labels
