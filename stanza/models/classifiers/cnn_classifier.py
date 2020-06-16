import logging
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import stanza.models.classifiers.classifier_args as classifier_args
from stanza.models.common.vocab import PAD_ID

logger = logging.getLogger('stanza')

class CNNClassifier(nn.Module):
    def __init__(self, emb_matrix, vocab, labels, args):
        """
        emb_matrix is a giant matrix of the pretrained embeddings

        labels is the list of labels we expect in the training data.
        Used to derive the number of classes.  Saving it in the model
        will let us check that test data has the same labels
        """
        super(CNNClassifier, self).__init__()
        self.labels = labels
        self.config = SimpleNamespace(filter_channels = args.filter_channels,
                                      filter_sizes = args.filter_sizes,
                                      fc_shapes = args.fc_shapes,
                                      dropout = args.dropout,
                                      num_classes = len(labels),
                                      wordvec_type = args.wordvec_type,
                                      model_type = 'CNNClassifier')

        self.unsaved_modules = []

        # TODO: make retraining vectors an option
        #   - note: this would greatly increase the size of the models
        #           we could make that an optional improvement, though
        #   - another alternative: make a set of delta vectors, possibly of
        #     lower dimension.  won't make the models too much bigger
        # TODO: freeze everything except pad & unk as an option
        self.add_unsaved_module('embedding', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        # The Pretrain has PAD and UNK already (indices 0 and 1), but we
        # possibly want to train UNK while freezing the rest of the embedding
        # note that the /10.0 operation has to be inside nn.Parameter unless
        # you want to spend a long time debugging this
        self.unk = nn.Parameter(torch.randn(self.embedding_dim) / np.sqrt(self.embedding_dim) / 10.0)

        self.vocab_map = { word: i for i, word in enumerate(vocab) }

        # Pytorch is "aware" of the existence of the nn.Modules inside
        # an nn.ModuleList in terms of parameters() etc
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1,
                                                    out_channels=self.config.filter_channels,
                                                    kernel_size=(filter_size, self.embedding_dim))
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

        # pad each phrase so either it matches the longest conv or the
        # longest phrase in the input, whichever is longer
        max_phrase_len = max(len(x) for x in inputs)
        if self.max_window > max_phrase_len:
            max_phrase_len = self.max_window

        batch_indices = []
        batch_unknowns = []
        for phrase in inputs:
            # TODO: random is good for train mode.  try something else at test time?
            begin_pad_width = random.randint(0, max_phrase_len - len(phrase))
            end_pad_width = max_phrase_len - begin_pad_width - len(phrase)
            sentence_indices = []
            sentence_unknowns = []
            for i in range(begin_pad_width):
                sentence_indices.append(PAD_ID)

            for word in phrase:
                # our current word vectors are all entirely lowercased
                word = word.lower()
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
            for i in range(end_pad_width):
                sentence_indices.append(PAD_ID)

            batch_indices.append(sentence_indices)
            batch_unknowns.append(sentence_unknowns)

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
    logger.info("Loaded model {}".format(filename))

    model_type = getattr(checkpoint['config'], 'model_type', 'CNNClassifier')
    if model_type == 'CNNClassifier':
        model = CNNClassifier(pretrain.emb, pretrain.vocab,
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
