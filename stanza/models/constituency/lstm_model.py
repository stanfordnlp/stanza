"""
A version of the BaseModel which uses LSTMs to predict the correct next transition
based on the current known state.

The primary purpose of this class is to implement the prediction of the next
transition, which is done by concatenating the output of an LSTM operated over
previous transitions, the words, and the partially built constituents.

A complete processing of a sentence is as follows:
  1) Run the input words through an encoder.
     The encoder includes some or all of the following:
       pretrained word embedding
       finetuned word embedding for training set words - "delta_embedding"
       POS tag embedding
       pretrained charlm representation
       BERT or similar large language model representation
       attention transformer over the previous inputs
       labeled attention transformer over the first attention layer
     The encoded input is then put through a bi-lstm, giving a word representation
  2) Transitions are put in an embedding, and transitions already used are tracked
     in an LSTM
  3) Constituents already built are also processed in an LSTM
  4) Every transition is chosen by taking the output of the current word position,
     the transition LSTM, and the constituent LSTM, and classifying the next
     transition
  5) Transitions are repeated (with constraints) until the sentence is completed
"""

from collections import namedtuple
from enum import Enum
import logging
import math
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from stanza.models.common.bert_embedding import extract_bert_embeddings
from stanza.models.common.utils import unsort
from stanza.models.common.vocab import PAD_ID, UNK_ID
from stanza.models.constituency.base_model import BaseModel
from stanza.models.constituency.label_attention import LabelAttentionModule
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.partitioned_transformer import PartitionedTransformerModule
from stanza.models.constituency.tree_stack import TreeStack
from stanza.models.constituency.utils import build_nonlinearity, initialize_linear, TextTooLongError

logger = logging.getLogger('stanza')

WordNode = namedtuple("WordNode", ['value', 'hx'])
TransitionNode = namedtuple("TransitionNode", ['value', 'output', 'lstm_hx', 'lstm_cx'])

# Invariant: the tree_hx at the top of the constituency stack will have a
# single dimension
# We do this to maintain consistency between the different operations,
# which sometimes result in different shapes
# This will be unsqueezed in order to put into the next layer if needed
# lstm_hx & lstm_cx are the hidden & cell states of the LSTM going across constituents
ConstituentNode = namedtuple("ConstituentNode", ['value', 'tree_hx', 'lstm_hx', 'lstm_cx'])
Constituent = namedtuple("Constituent", ['value', 'tree_hx'])

# The sentence boundary vectors are marginally useful at best.
# However, they make it much easier to use non-bert layers as input to
# attention layers, as the attention layers work better when they have
# an index 0 to attend to.
class SentenceBoundary(Enum):
    NONE               = 1
    WORDS              = 2
    EVERYTHING         = 3

# How to compose constituent children into new constituents
# MAX is simply take the max value of the children
# this is surprisingly effective
# for example, a Turkish dataset went from 81-81.5 dev, 75->75.5 test
# BILSTM is the method described in the papers of making an lstm
# out of the constituents
# BILSTM_MAX is the same as BILSTM, but instead of using a Linear
# to reduce the outputs of the lstm, we first take the max
# and then use a linear to reduce the max
# BIGRAM combines pairs of children and then takes the max over those
# ATTN means to put an attention layer over the children nodes
# we then take the max of the children with their attention
#
# Experiments show that MAX is noticeably better than the other options
# On ja_alt, here are a few results after 200 iterations,
# averaged over 5 iterations:
#   MAX:         0.8985
#   BILSTM:      0.8964
#   BILSTM_MAX:  0.8973
#   BIGRAM:      0.8982
#
# The MAX method has a linear transform after the max.
#   Removing that transform makes the score go down to 0.8982
#
# We tried a few varieties of BILSTM_MAX
# In particular:
# max over LSTM, combining forward & backward using the max: 0.8970
# max over forward & backward separately, then reduce:       0.8970
# max over forward & backward only over 1:-1
#   (eg, leave out the node embedding):                      0.8969
# same as previous, but split the reduce into 2 pieces:      0.8973
# max over forward & backward separately, then reduce as
#   1/2(F + B) + W(F,B)
#   the idea being that this way F and B are guaranteed
#   to be represented:                                       0.8971
#
# BIGRAM is an attempt to mix information from nodes
#   when building constituents, but it didn't help
#   The first example, just taking pairs and learning
#   a transform, went to NaN.  Likely the transform
#   expanded the embedding too much.  Switching it to
#   scale the matrix by 0.5 didn't go to Nan, but only
#   resulted in 0.8982
#
# A couple varieties of ATTN:
# first an input linear, then attn, then an output linear
#   the upside of this would be making the dimension of the attn
#   independent from the rest of the model
#   however, this caused an expansion in the magnitude of the vectors,
#   resulting in NaN for deep enough trees
# adding layernorm or tanh to balance this out resulted in
#   disappointing performance
#   tanh: 0.8972
# another alternative not tested yet: lower initialization weights
#   and enforce that the norms of the matrices are low enough that
#   exponential explosion up the layers of the tree doesn't happen
# just an attention layer means hidden_size % reduce_heads == 0
#   that is simple enough to enforce by slightly changing hidden_size
#   if needed
# appending the embedding for the open state to the start of the
#   sequence of children and taking only the content nodes
#   was very disappointing: 0.8967
# taking the entire sequence of children including the open state
#   embedding resulted in 0.8973
# long story short, this looks like an idea that should work, but it
#   doesn't help.  suggestions welcome for improving these results
class ConstituencyComposition(Enum):
    BILSTM                = 1
    MAX                   = 2
    BILSTM_MAX            = 4
    BIGRAM                = 5
    ATTN                  = 6

class LSTMModel(BaseModel, nn.Module):
    def __init__(self, pretrain, forward_charlm, backward_charlm, bert_model, bert_tokenizer, transitions, constituents, tags, words, rare_words, root_labels, constituent_opens, unary_limit, args):
        """
        pretrain: a Pretrain object
        transitions: a list of all possible transitions which will be
          used to build trees
        constituents: a list of all possible constituents in the treebank
        tags: a list of all possible tags in the treebank
        words: a list of all known words, used for a delta word embedding.
          note that there will be an attempt made to learn UNK words as well,
          and tags by themselves may help UNK words
        rare_words: a list of rare words, used to occasionally replace with UNK
        root_labels: probably ROOT, although apparently some treebanks like TOP
        constituent_opens: a list of all possible open nodes which will go on the stack
          - this might be different from constituents if there are nodes
            which represent multiple constituents at once
        args: hidden_size, transition_hidden_size, etc as gotten from
          constituency_parser.py

        Note that it might look like a hassle to pass all of this in
        when it can be collected directly from the trees themselves.
        However, that would only work at train time.  At eval or
        pipeline time we will load the lists from the saved model.
        """
        super().__init__(transition_scheme=args['transition_scheme'], unary_limit=unary_limit)

        self.args = args
        self.unsaved_modules = []

        emb_matrix = pretrain.emb
        self.add_unsaved_module('embedding', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))

        # replacing NBSP picks up a whole bunch of words for VI
        self.vocab_map = { word.replace('\xa0', ' '): i for i, word in enumerate(pretrain.vocab) }
        # precompute tensors for the word indices
        # the tensors should be put on the GPU if needed with a call to cuda()
        self.register_buffer('vocab_tensors', torch.tensor(range(len(pretrain.vocab)), requires_grad=False))
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        self.root_labels = sorted(list(root_labels))
        self.constituents = sorted(list(constituents))

        self.hidden_size = self.args['hidden_size']
        self.constituency_composition = self.args.get("constituency_composition", ConstituencyComposition.BILSTM)
        if self.constituency_composition == ConstituencyComposition.ATTN:
            self.reduce_heads = self.args['reduce_heads']
            if self.hidden_size % self.reduce_heads != 0:
                self.hidden_size = self.hidden_size + self.reduce_heads - (self.hidden_size % self.reduce_heads)

        self.transition_hidden_size = self.args['transition_hidden_size']
        self.tag_embedding_dim = self.args['tag_embedding_dim']
        self.transition_embedding_dim = self.args['transition_embedding_dim']
        self.delta_embedding_dim = self.args['delta_embedding_dim']

        self.word_input_size = self.embedding_dim + self.tag_embedding_dim + self.delta_embedding_dim

        if forward_charlm is not None:
            self.add_unsaved_module('forward_charlm', forward_charlm)
            self.word_input_size += self.forward_charlm.hidden_dim()
            if not forward_charlm.is_forward_lm:
                raise ValueError("Got a backward charlm as a forward charlm!")
        else:
            self.forward_charlm = None
        if backward_charlm is not None:
            self.add_unsaved_module('backward_charlm', backward_charlm)
            self.word_input_size += self.backward_charlm.hidden_dim()
            if backward_charlm.is_forward_lm:
                raise ValueError("Got a forward charlm as a backward charlm!")
        else:
            self.backward_charlm = None

        self.delta_words = sorted(set(words))
        self.delta_word_map = { word: i+2 for i, word in enumerate(self.delta_words) }
        assert PAD_ID == 0
        assert UNK_ID == 1
        # initialization is chosen based on the observed values of the norms
        # after several long training cycles
        # (this is true for other embeddings and embedding-like vectors as well)
        # the experiments show this slightly helps were done with
        # Adadelta and the correct initialization may be slightly
        # different for a different optimizer.
        # in fact, it is likely a scheme other than normal_ would
        # be better - the optimizer tends to learn the weights
        # rather close to 0 before learning in the direction it
        # actually wants to go
        self.delta_embedding = nn.Embedding(num_embeddings = len(self.delta_words)+2,
                                            embedding_dim = self.delta_embedding_dim,
                                            padding_idx = 0)
        nn.init.normal_(self.delta_embedding.weight, std=0.05)
        self.register_buffer('delta_tensors', torch.tensor(range(len(self.delta_words) + 2), requires_grad=False))

        self.rare_words = set(rare_words)

        self.tags = sorted(list(tags))
        if self.tag_embedding_dim > 0:
            self.tag_map = { t: i+2 for i, t in enumerate(self.tags) }
            self.tag_embedding = nn.Embedding(num_embeddings = len(tags)+2,
                                              embedding_dim = self.tag_embedding_dim,
                                              padding_idx = 0)
            nn.init.normal_(self.tag_embedding.weight, std=0.25)
            self.register_buffer('tag_tensors', torch.tensor(range(len(self.tags) + 2), requires_grad=False))

        self.num_lstm_layers = self.args['num_lstm_layers']
        self.lstm_layer_dropout = self.args['lstm_layer_dropout']

        # also register a buffer of zeros so that we can always get zeros on the appropriate device
        self.register_buffer('word_zeros', torch.zeros(self.hidden_size))
        self.register_buffer('transition_zeros', torch.zeros(self.num_lstm_layers, 1, self.transition_hidden_size))
        self.register_buffer('constituent_zeros', torch.zeros(self.num_lstm_layers, 1, self.hidden_size))

        # possibly add a couple vectors for bookends of the sentence
        # We put the word_start and word_end here, AFTER counting the
        # charlm dimension, but BEFORE counting the bert dimension,
        # as we want word_start and word_end to not have dimensions
        # for the bert embedding.  The bert model will add its own
        # start and end representation.
        self.sentence_boundary_vectors = self.args['sentence_boundary_vectors']
        if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
            self.register_parameter('word_start_embedding', torch.nn.Parameter(0.2 * torch.randn(self.word_input_size, requires_grad=True)))
            self.register_parameter('word_end_embedding', torch.nn.Parameter(0.2 * torch.randn(self.word_input_size, requires_grad=True)))

        # we set up the bert AFTER building word_start and word_end
        # so that we can use the charlm endpoint values rather than
        # try to train our own
        self.add_unsaved_module('bert_model', bert_model)
        self.add_unsaved_module('bert_tokenizer', bert_tokenizer)
        if bert_model is not None:
            if bert_tokenizer is None:
                raise ValueError("Cannot have a bert model without a tokenizer")
            self.bert_dim = self.bert_model.config.hidden_size
            self.word_input_size = self.word_input_size + self.bert_dim

        self.partitioned_transformer_module = None
        self.pattn_d_model = 0
        if LSTMModel.uses_pattn(self.args):
            # Initializations of parameters for the Partitioned Attention
            # round off the size of the model so that it divides in half evenly
            self.pattn_d_model = self.args['pattn_d_model'] // 2 * 2

            # Initializations for the Partitioned Attention
            # experiments suggest having a bias does not help here
            self.partitioned_transformer_module = PartitionedTransformerModule(
                self.args['pattn_num_layers'],
                d_model=self.pattn_d_model,
                n_head=self.args['pattn_num_heads'],
                d_qkv=self.args['pattn_d_kv'],
                d_ff=self.args['pattn_d_ff'],
                ff_dropout=self.args['pattn_relu_dropout'],
                residual_dropout=self.args['pattn_residual_dropout'],
                attention_dropout=self.args['pattn_attention_dropout'],
                word_input_size=self.word_input_size,
                bias=self.args['pattn_bias'],
                morpho_emb_dropout=self.args['pattn_morpho_emb_dropout'],
                timing=self.args['pattn_timing'],
                encoder_max_len=self.args['pattn_encoder_max_len']
            )
            self.word_input_size += self.pattn_d_model

        self.label_attention_module = None
        if LSTMModel.uses_lattn(self.args):
            if self.partitioned_transformer_module is None:
                logger.error("Not using Labeled Attention, as the Partitioned Attention module is not used")
            else:
                # TODO: think of a couple ways to use alternate inputs
                # for example, could pass in the word inputs with a positional embedding
                # that would also allow it to work in the case of no partitioned module
                if self.args['lattn_combined_input']:
                    self.lattn_d_input = self.word_input_size
                else:
                    self.lattn_d_input = self.pattn_d_model
                self.label_attention_module = LabelAttentionModule(self.lattn_d_input,
                                                                   self.args['lattn_d_input_proj'],
                                                                   self.args['lattn_d_kv'],
                                                                   self.args['lattn_d_kv'],
                                                                   self.args['lattn_d_l'],
                                                                   self.args['lattn_d_proj'],
                                                                   self.args['lattn_combine_as_self'],
                                                                   self.args['lattn_resdrop'],
                                                                   self.args['lattn_q_as_matrix'],
                                                                   self.args['lattn_residual_dropout'],
                                                                   self.args['lattn_attention_dropout'],
                                                                   self.pattn_d_model // 2,
                                                                   self.args['lattn_d_ff'],
                                                                   self.args['lattn_relu_dropout'],
                                                                   self.args['lattn_partitioned'])
                self.word_input_size = self.word_input_size + self.args['lattn_d_proj']*self.args['lattn_d_l']

        self.word_lstm = nn.LSTM(input_size=self.word_input_size, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers, bidirectional=True, dropout=self.lstm_layer_dropout)

        # after putting the word_delta_tag input through the word_lstm, we get back
        # hidden_size * 2 output with the front and back lstms concatenated.
        # this transforms it into hidden_size with the values mixed together
        self.word_to_constituent = nn.Linear(self.hidden_size * 2, self.hidden_size)
        initialize_linear(self.word_to_constituent, self.args['nonlinearity'], self.hidden_size * 2)

        self.transitions = sorted(list(transitions))
        self.transition_map = { t: i for i, t in enumerate(self.transitions) }
        # precompute tensors for the transitions
        self.register_buffer('transition_tensors', torch.tensor(range(len(transitions)), requires_grad=False))
        self.transition_embedding = nn.Embedding(num_embeddings = len(transitions),
                                                 embedding_dim = self.transition_embedding_dim)
        nn.init.normal_(self.transition_embedding.weight, std=0.25)
        if self.sentence_boundary_vectors is SentenceBoundary.EVERYTHING:
            self.register_parameter('transition_start_embedding', torch.nn.Parameter(0.2 * torch.randn(self.transition_hidden_size, requires_grad=True)))
        self.transition_lstm = nn.LSTM(input_size=self.transition_embedding_dim, hidden_size=self.transition_hidden_size, num_layers=self.num_lstm_layers, dropout=self.lstm_layer_dropout)

        self.constituent_opens = sorted(list(constituent_opens))
        # an embedding for the spot on the constituent LSTM taken up by the Open transitions
        # the pattern when condensing constituents is embedding - con1 - con2 - con3 - embedding
        # TODO: try the two ends have different embeddings?
        self.constituent_open_map = { x: i for (i, x) in enumerate(self.constituent_opens) }
        self.constituent_open_embedding = nn.Embedding(num_embeddings = len(self.constituent_open_map),
                                                       embedding_dim = self.hidden_size)
        nn.init.normal_(self.constituent_open_embedding.weight, std=0.2)
        if self.sentence_boundary_vectors is SentenceBoundary.EVERYTHING:
            self.register_parameter('constituent_start_embedding', torch.nn.Parameter(0.2 * torch.randn(self.hidden_size, requires_grad=True)))
        # input_size is hidden_size - could introduce a new constituent_size instead if we liked
        self.constituent_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers, dropout=self.lstm_layer_dropout)

        if args['combined_dummy_embedding']:
            self.dummy_embedding = self.constituent_open_embedding
        else:
            self.dummy_embedding = nn.Embedding(num_embeddings = len(self.constituent_open_map),
                                                embedding_dim = self.hidden_size)
            nn.init.normal_(self.dummy_embedding.weight, std=0.2)
        self.register_buffer('constituent_open_tensors', torch.tensor(range(len(constituent_opens)), requires_grad=False))

        # TODO: refactor
        if (self.constituency_composition == ConstituencyComposition.BILSTM or
            self.constituency_composition == ConstituencyComposition.BILSTM_MAX):
            # forward and backward pieces for crunching several
            # constituents into one, combined into a bi-lstm
            # TODO: make the hidden size here an option?
            self.constituent_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers, bidirectional=True, dropout=self.lstm_layer_dropout)
            # affine transformation from bi-lstm reduce to a new hidden layer
            if self.constituency_composition == ConstituencyComposition.BILSTM:
                self.reduce_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
                initialize_linear(self.reduce_linear, self.args['nonlinearity'], self.hidden_size * 2)
            else:
                self.reduce_forward = nn.Linear(self.hidden_size, self.hidden_size)
                self.reduce_backward = nn.Linear(self.hidden_size, self.hidden_size)
                initialize_linear(self.reduce_forward, self.args['nonlinearity'], self.hidden_size)
                initialize_linear(self.reduce_backward, self.args['nonlinearity'], self.hidden_size)
        elif self.constituency_composition == ConstituencyComposition.MAX:
            # transformation to turn several constituents into one new constituent
            self.reduce_linear = nn.Linear(self.hidden_size, self.hidden_size)
            initialize_linear(self.reduce_linear, self.args['nonlinearity'], self.hidden_size)
        elif self.constituency_composition == ConstituencyComposition.BIGRAM:
            self.reduce_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.reduce_bigram = nn.Linear(self.hidden_size * 2, self.hidden_size)
            initialize_linear(self.reduce_linear, self.args['nonlinearity'], self.hidden_size)
            initialize_linear(self.reduce_bigram, self.args['nonlinearity'], self.hidden_size)
        elif self.constituency_composition == ConstituencyComposition.ATTN:
            self.reduce_attn = nn.MultiheadAttention(self.hidden_size, self.reduce_heads)
        else:
            raise ValueError("Unhandled ConstituencyComposition: {}".format(self.constituency_composition))

        self.nonlinearity = build_nonlinearity(self.args['nonlinearity'])

        self.word_dropout = nn.Dropout(self.args['word_dropout'])
        self.predict_dropout = nn.Dropout(self.args['predict_dropout'])
        self.lstm_input_dropout = nn.Dropout(self.args['lstm_input_dropout'])

        # matrix for predicting the next transition using word/constituent/transition queues
        # word size + constituency size + transition size
        self.output_layers = self.build_output_layers(self.args['num_output_layers'], len(transitions))

    @staticmethod
    def uses_lattn(args):
        return args.get('lattn_d_proj', 0) > 0 and args.get('lattn_d_l', 0) > 0

    @staticmethod
    def uses_pattn(args):
        return args['pattn_num_heads'] > 0 and args['pattn_num_layers'] > 0

    def copy_with_new_structure(self, other):
        """
        Copy parameters from the other model to this model

        word_lstm can change size if the other model didn't use pattn / lattn and this one does.
        In that case, the new values are initialized to 0.
        This will rebuild the model in such a way that the outputs will be
        exactly the same as the previous model.
        """
        for name, other_parameter in other.named_parameters():
            if name.startswith('word_lstm.weight_ih_l0'):
                # bottom layer shape may have changed from adding a new pattn / lattn block
                my_parameter = self.get_parameter(name)
                copy_size = min(other_parameter.data.shape[1], my_parameter.data.shape[1])
                #new_values = my_parameter.data.clone().detach()
                new_values = torch.zeros_like(my_parameter.data)
                new_values[:, :copy_size] = other_parameter.data[:, :copy_size]
                my_parameter.data.copy_(new_values)
            else:
                self.get_parameter(name).data.copy_(other_parameter.data)

    def build_output_layers(self, num_output_layers, final_layer_size):
        """
        Build a ModuleList of Linear transformations for the given num_output_layers

        The final layer size can be specified.
        Initial layer size is the combination of word, constituent, and transition vectors
        Middle layer sizes are self.hidden_size
        """
        middle_layers = num_output_layers - 1
        # word_lstm:        hidden_size
        # transition_lstm:  transition_hidden_size
        # constituent_lstm: hidden_size
        predict_input_size = [self.hidden_size * 2 + self.transition_hidden_size] + [self.hidden_size] * middle_layers
        predict_output_size = [self.hidden_size] * middle_layers + [final_layer_size]
        output_layers = nn.ModuleList([nn.Linear(input_size, output_size)
                                       for input_size, output_size in zip(predict_input_size, predict_output_size)])
        for output_layer, input_size in zip(output_layers, predict_input_size):
            initialize_linear(output_layer, self.args['nonlinearity'], input_size)
        return output_layers

    def num_words_known(self, words):
        return sum(word in self.vocab_map or word.lower() in self.vocab_map for word in words)

    def add_unsaved_module(self, name, module):
        """
        Adds a module which will not be saved to disk

        Best used for large models such as pretrained word embeddings
        """
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def is_unsaved_module(self, name):
        return name.split('.')[0] in self.unsaved_modules

    def get_root_labels(self):
        return self.root_labels

    def log_norms(self):
        lines = ["NORMS FOR MODEL PARAMTERS"]
        for name, param in self.named_parameters():
            if param.requires_grad and name.split(".")[0] not in ('bert_model', 'forward_charlm', 'backward_charlm'):
                lines.append("%s %.6g" % (name, torch.norm(param).item()))
        logger.info("\n".join(lines))

    def initial_word_queues(self, tagged_word_lists):
        """
        Produce initial word queues out of the model's LSTMs for use in the tagged word lists.

        Operates in a batched fashion to reduce the runtime for the LSTM operations
        """
        device = next(self.parameters()).device

        vocab_map = self.vocab_map
        def map_word(word):
            idx = vocab_map.get(word, None)
            if idx is not None:
                return idx
            return vocab_map.get(word.lower(), UNK_ID)

        all_word_inputs = []
        all_word_labels = [[word.children[0].label for word in tagged_words]
                           for tagged_words in tagged_word_lists]

        for sentence_idx, tagged_words in enumerate(tagged_word_lists):
            word_labels = all_word_labels[sentence_idx]
            word_idx = torch.stack([self.vocab_tensors[map_word(word.children[0].label)] for word in tagged_words])
            word_input = self.embedding(word_idx)

            # this occasionally learns UNK at train time
            if self.training:
                delta_labels = [None if word in self.rare_words and random.random() < self.args['rare_word_unknown_frequency'] else word
                                for word in word_labels]
            else:
                delta_labels = word_labels
            delta_idx = torch.stack([self.delta_tensors[self.delta_word_map.get(word, UNK_ID)] for word in delta_labels])

            delta_input = self.delta_embedding(delta_idx)
            word_inputs = [word_input, delta_input]

            if self.tag_embedding_dim > 0:
                if self.training:
                    tag_labels = [None if random.random() < self.args['tag_unknown_frequency'] else word.label for word in tagged_words]
                else:
                    tag_labels = [word.label for word in tagged_words]
                tag_idx = torch.stack([self.tag_tensors[self.tag_map.get(tag, UNK_ID)] for tag in tag_labels])
                tag_input = self.tag_embedding(tag_idx)
                word_inputs.append(tag_input)

            all_word_inputs.append(word_inputs)

        if self.forward_charlm is not None:
            all_forward_chars = self.forward_charlm.build_char_representation(all_word_labels)
            for word_inputs, forward_chars in zip(all_word_inputs, all_forward_chars):
                word_inputs.append(forward_chars)
        if self.backward_charlm is not None:
            all_backward_chars = self.backward_charlm.build_char_representation(all_word_labels)
            for word_inputs, backward_chars in zip(all_word_inputs, all_backward_chars):
                word_inputs.append(backward_chars)

        all_word_inputs = [torch.cat(word_inputs, dim=1) for word_inputs in all_word_inputs]
        if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
            word_start = self.word_start_embedding.unsqueeze(0)
            word_end = self.word_end_embedding.unsqueeze(0)
            all_word_inputs = [torch.cat([word_start, word_inputs, word_end], dim=0) for word_inputs in all_word_inputs]

        if self.bert_model is not None:
            # BERT embedding extraction
            # result will be len+2 for each sentence
            # we will take 1:-1 if we don't care about the endpoints
            bert_embeddings = extract_bert_embeddings(self.args['bert_model'], self.bert_tokenizer, self.bert_model, all_word_labels, device,
                                                      keep_endpoints=self.sentence_boundary_vectors is not SentenceBoundary.NONE)
            all_word_inputs = [torch.cat((x, y), axis=1) for x, y in zip(all_word_inputs, bert_embeddings)]

        # Extract partitioned representation
        if self.partitioned_transformer_module is not None:
            partitioned_embeddings = self.partitioned_transformer_module(None, all_word_inputs)
            all_word_inputs = [torch.cat((x, y[:x.shape[0], :]), axis=1) for x, y in zip(all_word_inputs, partitioned_embeddings)]

        # Extract Labeled Representation
        if self.label_attention_module is not None:
            if self.args['lattn_combined_input']:
                labeled_representations = self.label_attention_module(all_word_inputs, tagged_word_lists)
            else:
                labeled_representations = self.label_attention_module(partitioned_embeddings, tagged_word_lists)
            all_word_inputs = [torch.cat((x, y[:x.shape[0], :]), axis=1) for x, y in zip(all_word_inputs, labeled_representations)]

        all_word_inputs = [self.word_dropout(word_inputs) for word_inputs in all_word_inputs]
        packed_word_input = torch.nn.utils.rnn.pack_sequence(all_word_inputs, enforce_sorted=False)
        word_output, _ = self.word_lstm(packed_word_input)
        # would like to do word_to_constituent here, but it seems PackedSequence doesn't support Linear
        # word_output will now be sentence x batch x 2*hidden_size
        word_output, word_output_lens = torch.nn.utils.rnn.pad_packed_sequence(word_output)
        # now sentence x batch x hidden_size

        word_queues = []
        for sentence_idx, tagged_words in enumerate(tagged_word_lists):
            if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
                sentence_output = word_output[1:len(tagged_words)+2, sentence_idx, :]
            else:
                sentence_output = word_output[:len(tagged_words), sentence_idx, :]
            sentence_output = self.word_to_constituent(sentence_output)
            sentence_output = self.nonlinearity(sentence_output)
            # TODO: this makes it so constituents downstream are
            # build with the outputs of the LSTM, not the word
            # embeddings themselves.  It is possible we want to
            # transform the word_input to hidden_size in some way
            # and use that instead
            if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
                word_queue = [WordNode(tag_node, sentence_output[idx, :])
                              for idx, tag_node in enumerate(tagged_words)]
                word_queue.append(WordNode(None, sentence_output[len(tagged_words), :]))
            else:
                word_queue = [WordNode(tag_node, sentence_output[idx, :])
                              for idx, tag_node in enumerate(tagged_words)]
                word_queue.append(WordNode(None, self.word_zeros))

            word_queues.append(word_queue)

        return word_queues

    def initial_transitions(self):
        """
        Return an initial TreeStack with no transitions

        Note that the transition_start operation is already batched, in a sense
        The subsequent batch built this way will be used for batch_size trees
        """
        if self.sentence_boundary_vectors is SentenceBoundary.EVERYTHING:
            transition_start = self.transition_start_embedding.unsqueeze(0).unsqueeze(0)
            output, (hx, cx) = self.transition_lstm(transition_start)
            transition_start = output[0, 0, :]
        else:
            transition_start = self.transition_zeros[-1, 0, :]
            hx = self.transition_zeros
            cx = self.transition_zeros
        return TreeStack(value=TransitionNode(None, transition_start, hx, cx), parent=None, length=1)

    def initial_constituents(self):
        """
        Return an initial TreeStack with no constituents
        """
        if self.sentence_boundary_vectors is SentenceBoundary.EVERYTHING:
            constituent_start = self.constituent_start_embedding.unsqueeze(0).unsqueeze(0)
            output, (hx, cx) = self.constituent_lstm(constituent_start)
            constituent_start = output[0, 0, :]
        else:
            constituent_start = self.constituent_zeros[-1, 0, :]
            hx = self.constituent_zeros
            cx = self.constituent_zeros
        return TreeStack(value=ConstituentNode(None, constituent_start, hx, cx), parent=None, length=1)

    def get_word(self, word_node):
        return word_node.value

    def transform_word_to_constituent(self, state):
        word_node = state.word_queue[state.word_position]
        word = word_node.value
        return Constituent(value=word, tree_hx=word_node.hx)

    def dummy_constituent(self, dummy):
        label = dummy.label
        open_index = self.constituent_open_tensors[self.constituent_open_map[label]]
        hx = self.dummy_embedding(open_index)
        return Constituent(value=dummy, tree_hx=hx)

    def unary_transform(self, constituents, labels):
        # TODO: this can be faster by stacking things
        top_constituent = constituents.value
        for label in reversed(labels):
            # double nested: the Constituent is in a list of just one child
            # and there is just one item in the list (hence the stacking comment)
            top_constituent = self.build_constituents([(label,)], [[top_constituent]])[0]
        return top_constituent

    def build_constituents(self, labels, children_lists):
        """
        Build new constituents with the given label from the list of children

        labels is a list of labels for each of the new nodes to construct
        children_lists is a list of children that go under each of the new nodes
        lists of each are used so that we can stack operations
        """
        node_hx = [[child.tree_hx for child in children] for children in children_lists]

        if (self.constituency_composition == ConstituencyComposition.BILSTM or
            self.constituency_composition == ConstituencyComposition.BILSTM_MAX):
            label_hx = [self.constituent_open_embedding(self.constituent_open_tensors[self.constituent_open_map[label]]) for label in labels]
            max_length = max(len(children) for children in children_lists)
            zeros = torch.zeros(self.hidden_size, device=label_hx[0].device)
            # weirdly, this is faster than using pack_sequence
            unpacked_hx = [[lhx] + nhx + [lhx] + [zeros] * (max_length - len(nhx)) for lhx, nhx in zip(label_hx, node_hx)]
            unpacked_hx = [self.lstm_input_dropout(torch.stack(nhx)) for nhx in unpacked_hx]
            packed_hx = torch.stack(unpacked_hx, axis=1)
            packed_hx = torch.nn.utils.rnn.pack_padded_sequence(packed_hx, [len(x)+2 for x in children_lists], enforce_sorted=False)
            lstm_output = self.constituent_reduce_lstm(packed_hx)
            # take just the output of the final layer
            #   result of lstm is ouput, (hx, cx)
            #   so [1][0] gets hx
            #      [1][0][-1] is the final output
            # will be shape len(children_lists) * 2, hidden_size for bidirectional
            # where forward outputs are -2 and backwards are -1
            if self.constituency_composition == ConstituencyComposition.BILSTM:
                lstm_output = lstm_output[1][0]
                forward_hx = lstm_output[-2, :, :]
                backward_hx = lstm_output[-1, :, :]
                hx = self.reduce_linear(torch.cat((forward_hx, backward_hx), axis=1))
            else:
                lstm_output, lstm_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_output[0])
                lstm_output = [lstm_output[1:length-1, x, :] for x, length in zip(range(len(lstm_lengths)), lstm_lengths)]
                lstm_output = torch.stack([torch.max(x, 0).values for x in lstm_output], axis=0)
                hx = self.reduce_forward(lstm_output[:, :self.hidden_size]) + self.reduce_backward(lstm_output[:, self.hidden_size:])
        elif self.constituency_composition == ConstituencyComposition.MAX:
            unpacked_hx = [self.lstm_input_dropout(torch.max(torch.stack(nhx), 0).values) for nhx in node_hx]
            packed_hx = torch.stack(unpacked_hx, axis=0)
            hx = self.reduce_linear(packed_hx)
        elif self.constituency_composition == ConstituencyComposition.BIGRAM:
            unpacked_hx = []
            for nhx in node_hx:
                # tanh or otherwise limit the size of the output?
                stacked_nhx = self.lstm_input_dropout(torch.stack(nhx))
                if stacked_nhx.shape[0] > 1:
                    bigram_hx = torch.cat((stacked_nhx[:-1, :], stacked_nhx[1:, :]), axis=1)
                    bigram_hx = self.reduce_bigram(bigram_hx) / 2
                    stacked_nhx = torch.cat((stacked_nhx, bigram_hx), axis=0)
                unpacked_hx.append(torch.max(stacked_nhx, 0).values)
            packed_hx = torch.stack(unpacked_hx, axis=0)
            hx = self.reduce_linear(packed_hx)
        elif self.constituency_composition == ConstituencyComposition.ATTN:
            label_hx = [self.constituent_open_embedding(self.constituent_open_tensors[self.constituent_open_map[label]]) for label in labels]
            unpacked_hx = [torch.stack(nhx).unsqueeze(1) for nhx in node_hx]
            unpacked_hx = [torch.cat((lhx.unsqueeze(0).unsqueeze(0), nhx), axis=0) for lhx, nhx in zip(label_hx, unpacked_hx)]
            unpacked_hx = [self.reduce_attn(nhx, nhx, nhx)[0].squeeze(1) for nhx in unpacked_hx]
            unpacked_hx = [self.lstm_input_dropout(torch.max(nhx, 0).values) for nhx in unpacked_hx]
            hx = torch.stack(unpacked_hx, axis=0)
        else:
            raise ValueError("Unhandled ConstituencyComposition: {}".format(self.constituency_composition))

        hx = self.nonlinearity(hx)

        constituents = []
        for idx, (label, children) in enumerate(zip(labels, children_lists)):
            children = [child.value for child in children]
            if isinstance(label, str):
                node = Tree(label=label, children=children)
            else:
                for value in reversed(label):
                    node = Tree(label=value, children=children)
                    children = node
            constituents.append(Constituent(value=node, tree_hx=hx[idx, :]))
        return constituents

    def push_constituents(self, constituent_stacks, constituents):
        current_nodes = [stack.value for stack in constituent_stacks]

        constituent_input = torch.stack([x.tree_hx for x in constituents])
        constituent_input = constituent_input.unsqueeze(0)
        constituent_input = self.lstm_input_dropout(constituent_input)

        hx = torch.cat([current_node.lstm_hx for current_node in current_nodes], axis=1)
        cx = torch.cat([current_node.lstm_cx for current_node in current_nodes], axis=1)
        output, (hx, cx) = self.constituent_lstm(constituent_input, (hx, cx))
        # Another possibility here would be to use output[0, i, :]
        # from the constituency lstm for the value of the new node.
        # This might theoretically make the new constituent include
        # information from neighboring constituents.  However, this
        # lowers the scores of various models.
        # For example, an experiment on ja_alt built this way,
        # averaged over 5 trials, had the following loss in accuracy:
        # 150 epochs: 0.8971 to 0.8953
        # 200 epochs: 0.8985 to 0.8964
        new_stacks = [stack.push(ConstituentNode(constituent.value, constituents[i].tree_hx, hx[:, i:i+1, :], cx[:, i:i+1, :]))
                      for i, (stack, constituent) in enumerate(zip(constituent_stacks, constituents))]
        return new_stacks

    def get_top_constituent(self, constituents):
        """
        Extract only the top constituent from a state's constituent
        sequence, even though it has multiple addition pieces of
        information
        """
        constituent_node = constituents.value
        return constituent_node.value

    def push_transitions(self, transition_stacks, transitions):
        """
        Push all of the given transitions on to the stack as a batch operations.

        Significantly faster than doing one transition at a time.
        """
        transition_idx = torch.stack([self.transition_tensors[self.transition_map[transition]] for transition in transitions])
        transition_input = self.transition_embedding(transition_idx).unsqueeze(0)
        transition_input = self.lstm_input_dropout(transition_input)

        hx = torch.cat([t.value.lstm_hx for t in transition_stacks], axis=1)
        cx = torch.cat([t.value.lstm_cx for t in transition_stacks], axis=1)
        output, (hx, cx) = self.transition_lstm(transition_input, (hx, cx))
        new_stacks = [stack.push(TransitionNode(transition, output[0, i, :], hx[:, i:i+1, :], cx[:, i:i+1, :]))
                      for i, (stack, transition) in enumerate(zip(transition_stacks, transitions))]
        return new_stacks

    def get_top_transition(self, transitions):
        """
        Extract only the top transition from a state's transition
        sequence, even though it has multiple addition pieces of
        information
        """
        transition_node = transitions.value
        return transition_node.value

    def forward(self, states):
        """
        Return logits for a prediction of what transition to make next

        We've basically done all the work analyzing the state as
        part of applying the transitions, so this method is very simple
        """
        word_hx = torch.stack([state.word_queue[state.word_position].hx for state in states])
        transition_hx = torch.stack([state.transitions.value.output for state in states])
        # note that we use hx instead of output from the constituents
        # this way, we can, as an option, NOT include the constituents to the left
        # when building the current vector for a constituent
        # and the vector used for inference will still incorporate the entire LSTM
        constituent_hx = torch.stack([state.constituents.value.lstm_hx[-1, 0, :] for state in states])

        hx = torch.cat((word_hx, transition_hx, constituent_hx), axis=1)
        for idx, output_layer in enumerate(self.output_layers):
            hx = self.predict_dropout(hx)
            if idx < len(self.output_layers) - 1:
                hx = self.nonlinearity(hx)
            hx = output_layer(hx)
        return hx

    def predict(self, states, is_legal=True):
        """
        Generate and return predictions, along with the transitions those predictions represent

        If is_legal is set to True, will only return legal transitions.
        This means returning None if there are no legal transitions.
        Hopefully the constraints prevent that from happening
        """
        predictions = self.forward(states)
        pred_max = torch.argmax(predictions, axis=1)

        pred_trans = [self.transitions[pred_max[idx]] for idx in range(len(states))]
        if is_legal:
            for idx, (state, trans) in enumerate(zip(states, pred_trans)):
                if not trans.is_legal(state, self):
                    _, indices = predictions[idx, :].sort(descending=True)
                    for index in indices:
                        if self.transitions[index].is_legal(state, self):
                            pred_trans[idx] = self.transitions[index]
                            break
                    else: # yeah, else on a for loop, deal with it
                        pred_trans[idx] = None

        return predictions, pred_trans

    def weighted_choice(self, states):
        """
        Generate and return predictions, and randomly choose a prediction weighted by the scores

        TODO: pass in a temperature
        """
        predictions = self.forward(states)
        pred_trans = []
        for state, prediction in zip(states, predictions):
            legal_idx = [idx for idx in range(prediction.shape[0]) if self.transitions[idx].is_legal(state, self)]
            if len(legal_idx) == 0:
                pred_trans.append(None)
                continue
            scores = prediction[legal_idx]
            scores = torch.softmax(scores, dim=0)
            idx = torch.multinomial(scores, 1)
            idx = legal_idx[idx]
            pred_trans.append(self.transitions[idx])
        return predictions, pred_trans

    def get_params(self, skip_modules=True):
        """
        Get a dictionary for saving the model
        """
        model_state = self.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if self.is_unsaved_module(k)]
            for k in skipped:
                del model_state[k]
        params = {
            'model': model_state,
            'model_type': "LSTM",
            'config': self.args,
            'transitions': self.transitions,
            'constituents': self.constituents,
            'tags': self.tags,
            'words': self.delta_words,
            'rare_words': self.rare_words,
            'root_labels': self.root_labels,
            'constituent_opens': self.constituent_opens,
            'unary_limit': self.unary_limit(),
        }

        return params

