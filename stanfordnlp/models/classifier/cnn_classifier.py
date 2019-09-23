import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, emb_matrix, vocab, num_classes, args):
        """
        emb_matrix is a giant matrix of the pretrained embeddings
        """
        super(CNNClassifier, self).__init__()
        self.config = SimpleNamespace(filter_channels = args.filter_channels,
                                      filter_sizes = args.filter_sizes,
                                      fc_shapes = args.fc_shapes,
                                      dropout = args.dropout,
                                      num_classes = num_classes)

        self.unsaved_modules = []

        # TODO: make retraining vectors an option
        # TODO: make this trans_pretrained as with the pos model?
        #   - we could train the trans matrix too
        #   - for 1 word phrases we could just use the trans matrix and
        #     label based on that
        # TODO: freeze everything except pad & unk as an option
        self.add_unsaved_module('embedding', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        # The Pretrain has PAD and UNK already (indices 0 and 1), but we
        # possibly want to train UNK while freezing the rest of the embedding
        self.pad = vocab[0]

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
        fc_layers.append(nn.Linear(previous_layer_size, num_classes))
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

        if max_phrase_len > min(len(x) for x in inputs):
            idx = torch.tensor(self.vocab_map[self.pad], requires_grad=False, device=device)
            pad_vector = self.embedding(idx)

        input_tensor = []
        for phrase in inputs:
            # build a list of the vectors we want for this sentence / phrase
            input_vectors = []
            # TODO: random is good for train mode.  try something else at test time?
            begin_pad_width = random.randint(0, max_phrase_len - len(phrase))
            end_pad_width = max_phrase_len - begin_pad_width - len(phrase)
            for i in range(begin_pad_width):
                input_vectors.append(pad_vector)

            for word in phrase:
                # our current word vectors are all entirely lowercased
                word = word.lower()
                if word in self.vocab_map:
                    idx = torch.tensor(self.vocab_map[word], requires_grad=False, device=device)
                    input_vectors.append(self.embedding(idx))
                    continue
                new_word = word.replace("-", "")
                # google vectors have words which are all dashes
                if len(new_word) == 0:
                    new_word = word
                if new_word in self.vocab_map:
                    idx = torch.tensor(self.vocab_map[new_word], requires_grad=False, device=device)
                    input_vectors.append(self.embedding(idx))
                    continue

                if new_word[-1] == "'":
                    new_word = new_word[:-1]
                    if new_word in self.vocab_map:
                        idx = torch.tensor(self.vocab_map[new_word], requires_grad=False, device=device)
                        input_vectors.append(self.embedding(idx))
                        continue
 
                # TODO: split UNK based on part of speech?  might be an interesting experiment
                input_vectors.append(self.unk)
            for i in range(end_pad_width):
                input_vectors.append(pad_vector)

            # we will now have an N x emb_size tensor
            # this is the input to the CNN
            # there are two ways in which this padding is suboptimal
            # the first is that for short sentences, smaller windows will
            #   be padded to the point that some windows are entirely pad
            # the second is that a sentence S will have more or less padding
            #   depending on what other sentences are in its batch
            # we assume these effects are pretty minimal
            x = torch.stack(input_vectors)

            # reshape x to 1xNxE
            x = x.unsqueeze(0)
            input_tensor.append(x)
        x = torch.stack(input_tensor)

        conv_outs = [self.dropout(F.relu(conv(x).squeeze(3)))
                     for conv in self.conv_layers]
        pool_outs = [F.max_pool1d(out, out.shape[2]).squeeze(2) for out in conv_outs]
        pooled = torch.cat(pool_outs, dim=1)

        previous_layer = pooled
        for fc in self.fc_layers[:-1]:
            previous_layer = self.dropout(F.relu(fc(previous_layer)))
        out = self.fc_layers[-1](previous_layer)
        return out
        

