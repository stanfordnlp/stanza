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
import copy
from enum import Enum
import logging
import math
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from stanza.models.common.bert_embedding import extract_bert_embeddings
from stanza.models.common.maxout_linear import MaxoutLinear
from stanza.models.common.utils import attach_bert_model, unsort
from stanza.models.common.vocab import PAD_ID, UNK_ID
from stanza.models.constituency.base_model import BaseModel
from stanza.models.constituency.label_attention import LabelAttentionModule
from stanza.models.constituency.lstm_tree_stack import LSTMTreeStack
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.partitioned_transformer import PartitionedTransformerModule
from stanza.models.constituency.positional_encoding import ConcatSinusoidalEncoding
from stanza.models.constituency.transformer_tree_stack import TransformerTreeStack
from stanza.models.constituency.tree_stack import TreeStack
from stanza.models.constituency.utils import build_nonlinearity, initialize_linear

logger = logging.getLogger('stanza')
tlogger = logging.getLogger('stanza.constituency.trainer')

WordNode = namedtuple("WordNode", ['value', 'hx'])

# lstm_hx & lstm_cx are the hidden & cell states of the LSTM going across constituents
# tree_hx and tree_cx are the states of the lstm going up the constituents in the case of the tree_lstm combination method
Constituent = namedtuple("Constituent", ['value', 'tree_hx', 'tree_cx'])

# The sentence boundary vectors are marginally useful at best.
# However, they make it much easier to use non-bert layers as input to
# attention layers, as the attention layers work better when they have
# an index 0 to attend to.
class SentenceBoundary(Enum):
    NONE               = 1
    WORDS              = 2
    EVERYTHING         = 3

class StackHistory(Enum):
    LSTM               = 1
    ATTN               = 2

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
#
# The current TREE_LSTM_CX mechanism uses a word's embedding
#   as the hx and a trained embedding over tags as the cx    0.8996
# This worked slightly better than 0s for cx (TREE_LSTM)     0.8992
# A variant of TREE_LSTM which didn't work out:
#   nodes are combined with an LSTM
#   hx & cx are embeddings of the node type (eg S, NP, etc)
#   input is the max over children:                          0.8977
# Another variant which didn't work: use the word embedding
#   as input to the same LSTM to get hx & cx                 0.8985
# Note that although the scores for TREE_LSTM_CX are slightly higher
# than MAX for the JA dataset, the benefit was not as clear for EN,
# so we left the default at MAX.
# For example, on English WSJ, before switching to Bert POS and
# a learned Bert mixing layer, a comparison of 5x models trained
# for 400 iterations got dev scores of:
#   TREE_LSTM_CX        0.9589
#   MAX                 0.9593
#
# UNTIED_MAX has a different reduce_linear for each type of
#   constituent in the model.  Similar to the different linear
#   maps used in the CVG paper from Socher, Bauer, Manning, Ng
# This is implemented as a large CxHxH parameter,
#   with num_constituent layers of hidden-hidden transform,
#   along with a CxH bias parameter.
#   Essentially C Linears stacked on top of each other,
#   but in a parameter so that indexing can be done quickly.
# Unfortunately this does not beat out MAX with one combined linear.
#   On an experiment on WSJ with all the best settings as of early
#   October 2022, such as a Bert model POS tagger:
#   MAX                 0.9597
#   UNTIED_MAX          0.9592
# Furthermore, starting from a finished MAX model and restarting
#   by splitting the MAX layer into multiple pieces did not improve.
#
# KEY has a single Key which is used for a facsimile of ATTN
#   each incoming subtree has its values weighted by a Query
#   then the Key is used to calculate a softmax
#   finally, a Value is used to scale the subtrees
#   reduce_heads is used to determine the number of heads
# There is an option to use or not use position information
#   using a sinusoidal position embedding
# UNTIED_KEY is the same, but has a different key
#   for each possible constituent
# On a VI dataset:
#   MAX                    0.82064
#   KEY (pos, 8)           0.81739
#   UNTIED_KEY (pos, 8)    0.82046
#   UNTIED_KEY (pos, 4)    0.81742
# Attempted to add a linear to mix the attn heads together,
#   but that was awful:    0.81567
# Adding two position vectors, one in each direction, did not help:
#   UNTIED_KEY (2x pos, 8) 0.8188
# To redo that experiment, double the width of reduce_query and
#   reduce_value, then call reduce_position on nhx, flip it,
#   and call reduce_position again
# Evidently the experiments to try should be:
#   no pos at all
#   more heads
class ConstituencyComposition(Enum):
    BILSTM                = 1
    MAX                   = 2
    TREE_LSTM             = 3
    BILSTM_MAX            = 4
    BIGRAM                = 5
    ATTN                  = 6
    TREE_LSTM_CX          = 7
    UNTIED_MAX            = 8
    KEY                   = 9
    UNTIED_KEY            = 10

class LSTMModel(BaseModel, nn.Module):
    def __init__(self, pretrain, forward_charlm, backward_charlm, bert_model, bert_tokenizer, force_bert_saved, peft_name, transitions, constituents, tags, words, rare_words, root_labels, constituent_opens, unary_limit, args):
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
        root_labels: probably ROOT, although apparently some treebanks like TOP or even s
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
        super().__init__(transition_scheme=args['transition_scheme'], unary_limit=unary_limit, reverse_sentence=args.get('reversed', False), root_labels=root_labels)

        self.args = args
        self.unsaved_modules = []

        emb_matrix = pretrain.emb
        self.add_unsaved_module('embedding', nn.Embedding.from_pretrained(emb_matrix, freeze=True))

        # replacing NBSP picks up a whole bunch of words for VI
        self.vocab_map = { word.replace('\xa0', ' '): i for i, word in enumerate(pretrain.vocab) }
        # precompute tensors for the word indices
        # the tensors should be put on the GPU if needed by calling to(device)
        self.register_buffer('vocab_tensors', torch.tensor(range(len(pretrain.vocab)), requires_grad=False))
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        self.constituents = sorted(list(constituents))

        self.hidden_size = self.args['hidden_size']
        self.constituency_composition = self.args.get("constituency_composition", ConstituencyComposition.BILSTM)
        if self.constituency_composition in (ConstituencyComposition.ATTN, ConstituencyComposition.KEY, ConstituencyComposition.UNTIED_KEY):
            self.reduce_heads = self.args['reduce_heads']
            if self.hidden_size % self.reduce_heads != 0:
                self.hidden_size = self.hidden_size + self.reduce_heads - (self.hidden_size % self.reduce_heads)

        if args['constituent_stack'] == StackHistory.ATTN:
            self.reduce_heads = self.args['reduce_heads']
            if self.hidden_size % args['constituent_heads'] != 0:
                # TODO: technically we should either use the LCM of this and reduce_heads, or just have two separate fields
                self.hidden_size = self.hidden_size + args['constituent_heads'] - (hidden_size % args['constituent_heads'])
                if self.constituency_composition == ConstituencyComposition.ATTN and self.hidden_size % self.reduce_heads != 0:
                    raise ValueError("--reduce_heads and --constituent_heads not compatible!")

        self.transition_hidden_size = self.args['transition_hidden_size']
        if args['transition_stack'] == StackHistory.ATTN:
            if self.transition_hidden_size % args['transition_heads'] > 0:
                logger.warning("transition_hidden_size %d %% transition_heads %d != 0.  reconfiguring", transition_hidden_size, args['transition_heads'])
                self.transition_hidden_size = self.transition_hidden_size + args['transition_heads'] - (self.transition_hidden_size % args['transition_heads'])

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
        self.num_tree_lstm_layers = self.args['num_tree_lstm_layers']
        self.lstm_layer_dropout = self.args['lstm_layer_dropout']

        self.word_dropout = nn.Dropout(self.args['word_dropout'])
        self.predict_dropout = nn.Dropout(self.args['predict_dropout'])
        self.lstm_input_dropout = nn.Dropout(self.args['lstm_input_dropout'])

        # also register a buffer of zeros so that we can always get zeros on the appropriate device
        self.register_buffer('word_zeros', torch.zeros(self.hidden_size * self.num_tree_lstm_layers))
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
        self.force_bert_saved = force_bert_saved or self.args['bert_finetune'] or self.args['stage1_bert_finetune']
        attach_bert_model(self, bert_model, bert_tokenizer, self.args.get('use_peft', False), self.force_bert_saved)
        self.peft_name = peft_name

        if bert_model is not None:
            if bert_tokenizer is None:
                raise ValueError("Cannot have a bert model without a tokenizer")
            self.bert_dim = self.bert_model.config.hidden_size
            if args['bert_hidden_layers']:
                # The average will be offset by 1/N so that the default zeros
                # represents an average of the N layers
                if args['bert_hidden_layers'] > bert_model.config.num_hidden_layers:
                    # limit ourselves to the number of layers actually available
                    # note that we can +1 because of the initial embedding layer
                    args['bert_hidden_layers'] = bert_model.config.num_hidden_layers + 1
                self.bert_layer_mix = nn.Linear(args['bert_hidden_layers'], 1, bias=False)
                nn.init.zeros_(self.bert_layer_mix.weight)
            else:
                # an average of layers 2, 3, 4 will be used
                # (for historic reasons)
                self.bert_layer_mix = None
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
        self.word_to_constituent = nn.Linear(self.hidden_size * 2, self.hidden_size * self.num_tree_lstm_layers)
        initialize_linear(self.word_to_constituent, self.args['nonlinearity'], self.hidden_size * 2)

        self.transitions = sorted(list(transitions))
        self.transition_map = { t: i for i, t in enumerate(self.transitions) }
        # precompute tensors for the transitions
        self.register_buffer('transition_tensors', torch.tensor(range(len(transitions)), requires_grad=False))
        self.transition_embedding = nn.Embedding(num_embeddings = len(transitions),
                                                 embedding_dim = self.transition_embedding_dim)
        nn.init.normal_(self.transition_embedding.weight, std=0.25)
        if args['transition_stack'] == StackHistory.LSTM:
            self.transition_stack = LSTMTreeStack(input_size=self.transition_embedding_dim,
                                                  hidden_size=self.transition_hidden_size,
                                                  num_lstm_layers=self.num_lstm_layers,
                                                  dropout=self.lstm_layer_dropout,
                                                  uses_boundary_vector=self.sentence_boundary_vectors is SentenceBoundary.EVERYTHING,
                                                  input_dropout=self.lstm_input_dropout)
        elif args['transition_stack'] == StackHistory.ATTN:
            self.transition_stack = TransformerTreeStack(input_size=self.transition_embedding_dim,
                                                         output_size=self.transition_hidden_size,
                                                         input_dropout=self.lstm_input_dropout,
                                                         use_position=True,
                                                         num_heads=args['transition_heads'])
        else:
            raise ValueError("Unhandled transition_stack StackHistory: {}".format(args['transition_stack']))

        self.constituent_opens = sorted(list(constituent_opens))
        # an embedding for the spot on the constituent LSTM taken up by the Open transitions
        # the pattern when condensing constituents is embedding - con1 - con2 - con3 - embedding
        # TODO: try the two ends have different embeddings?
        self.constituent_open_map = { x: i for (i, x) in enumerate(self.constituent_opens) }
        self.constituent_open_embedding = nn.Embedding(num_embeddings = len(self.constituent_open_map),
                                                       embedding_dim = self.hidden_size)
        nn.init.normal_(self.constituent_open_embedding.weight, std=0.2)

        # input_size is hidden_size - could introduce a new constituent_size instead if we liked
        if args['constituent_stack'] == StackHistory.LSTM:
            self.constituent_stack = LSTMTreeStack(input_size=self.hidden_size,
                                                   hidden_size=self.hidden_size,
                                                   num_lstm_layers=self.num_lstm_layers,
                                                   dropout=self.lstm_layer_dropout,
                                                   uses_boundary_vector=self.sentence_boundary_vectors is SentenceBoundary.EVERYTHING,
                                                   input_dropout=self.lstm_input_dropout)
        elif args['constituent_stack'] == StackHistory.ATTN:
            self.constituent_stack = TransformerTreeStack(input_size=self.hidden_size,
                                                          output_size=self.hidden_size,
                                                          input_dropout=self.lstm_input_dropout,
                                                          use_position=True,
                                                          num_heads=args['constituent_heads'])
        else:
            raise ValueError("Unhandled constituent_stack StackHistory: {}".format(args['transition_stack']))


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
        elif self.constituency_composition == ConstituencyComposition.UNTIED_MAX:
            # transformation to turn several constituents into one new constituent
            self.register_parameter('reduce_linear_weight', torch.nn.Parameter(torch.randn(len(constituent_opens), self.hidden_size, self.hidden_size, requires_grad=True)))
            self.register_parameter('reduce_linear_bias', torch.nn.Parameter(torch.randn(len(constituent_opens), self.hidden_size, requires_grad=True)))
            for layer_idx in range(len(constituent_opens)):
                nn.init.kaiming_normal_(self.reduce_linear_weight[layer_idx], nonlinearity=self.args['nonlinearity'])
            nn.init.uniform_(self.reduce_linear_bias, 0, 1 / (self.hidden_size * 2) ** 0.5)
        elif self.constituency_composition == ConstituencyComposition.BIGRAM:
            self.reduce_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.reduce_bigram = nn.Linear(self.hidden_size * 2, self.hidden_size)
            initialize_linear(self.reduce_linear, self.args['nonlinearity'], self.hidden_size)
            initialize_linear(self.reduce_bigram, self.args['nonlinearity'], self.hidden_size)
        elif self.constituency_composition == ConstituencyComposition.ATTN:
            self.reduce_attn = nn.MultiheadAttention(self.hidden_size, self.reduce_heads)
        elif self.constituency_composition == ConstituencyComposition.KEY or self.constituency_composition == ConstituencyComposition.UNTIED_KEY:
            if self.args['reduce_position']:
                # unsaved module so that if it grows, we don't save
                # the larger version unnecessarily
                # under any normal circumstances, the growth will
                # happen early in training when the model is not
                # behaving well, then will not be needed once the
                # model learns not to make super degenerate
                # constituents
                self.add_unsaved_module("reduce_position", ConcatSinusoidalEncoding(self.args['reduce_position'], 50))
            else:
                self.add_unsaved_module("reduce_position", nn.Identity())
            self.reduce_query = nn.Linear(self.hidden_size + self.args['reduce_position'], self.hidden_size, bias=False)
            self.reduce_value = nn.Linear(self.hidden_size + self.args['reduce_position'], self.hidden_size)
            if self.constituency_composition == ConstituencyComposition.KEY:
                self.register_parameter('reduce_key', torch.nn.Parameter(torch.randn(self.reduce_heads, self.hidden_size // self.reduce_heads, 1, requires_grad=True)))
            else:
                self.register_parameter('reduce_key', torch.nn.Parameter(torch.randn(len(constituent_opens), self.reduce_heads, self.hidden_size // self.reduce_heads, 1, requires_grad=True)))
        elif self.constituency_composition == ConstituencyComposition.TREE_LSTM:
            self.constituent_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_tree_lstm_layers, dropout=self.lstm_layer_dropout)
        elif self.constituency_composition == ConstituencyComposition.TREE_LSTM_CX:
            self.constituent_reduce_embedding = nn.Embedding(num_embeddings = len(tags)+2,
                                                             embedding_dim = self.num_tree_lstm_layers * self.hidden_size)
            self.constituent_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_tree_lstm_layers, dropout=self.lstm_layer_dropout)
        else:
            raise ValueError("Unhandled ConstituencyComposition: {}".format(self.constituency_composition))

        self.nonlinearity = build_nonlinearity(self.args['nonlinearity'])

        # matrix for predicting the next transition using word/constituent/transition queues
        # word size + constituency size + transition size
        # TODO: .get() is only necessary until all models rebuilt with this param
        self.maxout_k = self.args.get('maxout_k', 0)
        self.output_layers = self.build_output_layers(self.args['num_output_layers'], len(transitions), self.maxout_k)

    @staticmethod
    def uses_lattn(args):
        return args.get('use_lattn', True) and args.get('lattn_d_proj', 0) > 0 and args.get('lattn_d_l', 0) > 0

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
        if self.constituency_composition != other.constituency_composition and self.constituency_composition != ConstituencyComposition.UNTIED_MAX:
            raise ValueError("Models are incompatible: self.constituency_composition == {}, other.constituency_composition == {}".format(self.constituency_composition, other.constituency_composition))
        for name, other_parameter in other.named_parameters():
            # this allows other.constituency_composition == UNTIED_MAX to fall through
            if name.startswith('reduce_linear.') and self.constituency_composition == ConstituencyComposition.UNTIED_MAX:
                if name == 'reduce_linear.weight':
                    my_parameter = self.reduce_linear_weight
                elif name == 'reduce_linear.bias':
                    my_parameter = self.reduce_linear_bias
                else:
                    raise ValueError("Unexpected other parameter name {}".format(name))
                for idx in range(len(self.constituent_opens)):
                    my_parameter[idx].data.copy_(other_parameter.data)
            elif name.startswith('word_lstm.weight_ih_l0'):
                # bottom layer shape may have changed from adding a new pattn / lattn block
                my_parameter = self.get_parameter(name)
                # -1 so that it can be converted easier to a different parameter
                copy_size = min(other_parameter.data.shape[-1], my_parameter.data.shape[-1])
                #new_values = my_parameter.data.clone().detach()
                new_values = torch.zeros_like(my_parameter.data)
                new_values[..., :copy_size] = other_parameter.data[..., :copy_size]
                my_parameter.data.copy_(new_values)
            else:
                try:
                    self.get_parameter(name).data.copy_(other_parameter.data)
                except AttributeError as e:
                    raise AttributeError("Could not process %s" % name) from e

    def build_output_layers(self, num_output_layers, final_layer_size, maxout_k):
        """
        Build a ModuleList of Linear transformations for the given num_output_layers

        The final layer size can be specified.
        Initial layer size is the combination of word, constituent, and transition vectors
        Middle layer sizes are self.hidden_size
        """
        middle_layers = num_output_layers - 1
        # word_lstm:         hidden_size * num_tree_lstm_layers
        # transition_stack:  transition_hidden_size
        # constituent_stack: hidden_size
        predict_input_size = [self.hidden_size + self.hidden_size * self.num_tree_lstm_layers + self.transition_hidden_size] + [self.hidden_size] * middle_layers
        predict_output_size = [self.hidden_size] * middle_layers + [final_layer_size]
        if not maxout_k:
            output_layers = nn.ModuleList([nn.Linear(input_size, output_size)
                                           for input_size, output_size in zip(predict_input_size, predict_output_size)])
            for output_layer, input_size in zip(output_layers, predict_input_size):
                initialize_linear(output_layer, self.args['nonlinearity'], input_size)
        else:
            output_layers = nn.ModuleList([MaxoutLinear(input_size, output_size, maxout_k)
                                           for input_size, output_size in zip(predict_input_size, predict_output_size)])
        return output_layers

    def num_words_known(self, words):
        return sum(word in self.vocab_map or word.lower() in self.vocab_map for word in words)

    @property
    def retag_method(self):
        # TODO: make the method an enum
        return self.args['retag_method']

    def uses_xpos(self):
        return self.args['retag_package'] is not None and self.args['retag_method'] == 'xpos'

    def add_unsaved_module(self, name, module):
        """
        Adds a module which will not be saved to disk

        Best used for large models such as pretrained word embeddings
        """
        self.unsaved_modules += [name]
        setattr(self, name, module)
        if module is not None and name in ('forward_charlm', 'backward_charlm'):
            for _, parameter in module.named_parameters():
                parameter.requires_grad = False

    def is_unsaved_module(self, name):
        return name.split('.')[0] in self.unsaved_modules

    def get_norms(self):
        lines = []
        skip = set()
        if self.constituency_composition == ConstituencyComposition.UNTIED_MAX:
            skip = {'reduce_linear_weight', 'reduce_linear_bias'}
            lines.append("reduce_linear:")
            for c_idx, c_open in enumerate(self.constituent_opens):
                lines.append("  %s weight %.6g bias %.6g" % (c_open, torch.norm(self.reduce_linear_weight[c_idx]).item(), torch.norm(self.reduce_linear_bias[c_idx]).item()))
        active_params = [(name, param) for name, param in self.named_parameters() if param.requires_grad and name not in skip]
        if len(active_params) == 0:
            return lines
        print(len(active_params))

        max_name_len = max(len(name) for name, param in active_params)
        max_norm_len = max(len("%.6g" % torch.norm(param).item()) for name, param in active_params)
        format_string = "%-" + str(max_name_len) + "s   norm %" + str(max_norm_len) + "s  zeros %d / %d"
        for name, param in active_params:
            zeros = torch.sum(param.abs() < 0.000001).item()
            norm = "%.6g" % torch.norm(param).item()
            lines.append(format_string % (name, norm, zeros, param.nelement()))
        return lines

    def log_norms(self):
        lines = ["NORMS FOR MODEL PARAMETERS"]
        lines.extend(self.get_norms())
        logger.info("\n".join(lines))

    def log_shapes(self):
        lines = ["NORMS FOR MODEL PARAMETERS"]
        for name, param in self.named_parameters():
            if param.requires_grad:
                lines.append("{} {}".format(name, param.shape))
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
                                                      keep_endpoints=self.sentence_boundary_vectors is not SentenceBoundary.NONE,
                                                      num_layers=self.bert_layer_mix.in_features if self.bert_layer_mix is not None else None,
                                                      detach=not self.args['bert_finetune'] and not self.args['stage1_bert_finetune'],
                                                      peft_name=self.peft_name)
            if self.bert_layer_mix is not None:
                # add the average so that the default behavior is to
                # take an average of the N layers, and anything else
                # other than that needs to be learned
                bert_embeddings = [self.bert_layer_mix(feature).squeeze(2) + feature.sum(axis=2) / self.bert_layer_mix.in_features for feature in bert_embeddings]

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
                sentence_output = word_output[:len(tagged_words)+2, sentence_idx, :]
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
                word_queue =  [WordNode(None, sentence_output[0, :])]
                word_queue += [WordNode(tag_node, sentence_output[idx+1, :])
                               for idx, tag_node in enumerate(tagged_words)]
                word_queue.append(WordNode(None, sentence_output[len(tagged_words)+1, :]))
            else:
                word_queue =  [WordNode(None, self.word_zeros)]
                word_queue += [WordNode(tag_node, sentence_output[idx, :])
                                   for idx, tag_node in enumerate(tagged_words)]
                word_queue.append(WordNode(None, self.word_zeros))

            if self.reverse_sentence:
                word_queue = list(reversed(word_queue))
            word_queues.append(word_queue)

        return word_queues

    def initial_transitions(self):
        """
        Return an initial TreeStack with no transitions
        """
        return self.transition_stack.initial_state()

    def initial_constituents(self):
        """
        Return an initial TreeStack with no constituents
        """
        return self.constituent_stack.initial_state(Constituent(None, self.constituent_zeros, self.constituent_zeros))

    def get_word(self, word_node):
        return word_node.value

    def transform_word_to_constituent(self, state):
        word_node = state.get_word(state.word_position)
        word = word_node.value
        if self.constituency_composition == ConstituencyComposition.TREE_LSTM:
            return Constituent(word, word_node.hx.view(self.num_tree_lstm_layers, self.hidden_size), self.word_zeros.view(self.num_tree_lstm_layers, self.hidden_size))
        elif self.constituency_composition == ConstituencyComposition.TREE_LSTM_CX:
            # the UNK tag will be trained thanks to occasionally dropping out tags
            tag = word.label
            tree_hx = word_node.hx.view(self.num_tree_lstm_layers, self.hidden_size)
            tag_tensor = self.tag_tensors[self.tag_map.get(tag, UNK_ID)]
            tree_cx = self.constituent_reduce_embedding(tag_tensor)
            tree_cx = tree_cx.view(self.num_tree_lstm_layers, self.hidden_size)
            return Constituent(word, tree_hx, tree_cx * tree_hx)
        else:
            return Constituent(word, word_node.hx[:self.hidden_size].unsqueeze(0), None)

    def dummy_constituent(self, dummy):
        label = dummy.label
        open_index = self.constituent_open_tensors[self.constituent_open_map[label]]
        hx = self.dummy_embedding(open_index)
        # the cx doesn't matter: the dummy will be discarded when building a new constituent
        return Constituent(dummy, hx.unsqueeze(0), None)

    def build_constituents(self, labels, children_lists):
        """
        Build new constituents with the given label from the list of children

        labels is a list of labels for each of the new nodes to construct
        children_lists is a list of children that go under each of the new nodes
        lists of each are used so that we can stack operations
        """
        # at the end of each of these operations, we expect lstm_hx.shape
        # is (L, N, hidden_size) for N lists of children
        if (self.constituency_composition == ConstituencyComposition.BILSTM or
            self.constituency_composition == ConstituencyComposition.BILSTM_MAX):
            node_hx = [[child.value.tree_hx.squeeze(0) for child in children] for children in children_lists]
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
            lstm_hx = self.nonlinearity(hx).unsqueeze(0)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.MAX:
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            unpacked_hx = [self.lstm_input_dropout(torch.max(torch.stack(nhx), 0).values) for nhx in node_hx]
            packed_hx = torch.stack(unpacked_hx, axis=1)
            hx = self.reduce_linear(packed_hx)
            lstm_hx = self.nonlinearity(hx)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.UNTIED_MAX:
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            unpacked_hx = [self.lstm_input_dropout(torch.max(torch.stack(nhx), 0).values) for nhx in node_hx]
            # shape == len(labels),1,hidden_size after the stack
            #packed_hx = torch.stack(unpacked_hx, axis=0)
            label_indices = [self.constituent_open_map[label] for label in labels]
            # we would like to stack the reduce_linear_weight calculations as follows:
            #reduce_weight = self.reduce_linear_weight[label_indices]
            #reduce_bias = self.reduce_linear_bias[label_indices]
            # this would allow for faster vectorized operations.
            # however, this runs out of memory on larger training examples,
            # presumably because there are too many stacks in a row and each one
            # has its own gradient kept for the entire calculation
            # fortunately, this operation is not a huge part of the expense
            hx = [torch.matmul(self.reduce_linear_weight[label_idx], hx_layer.squeeze(0)) + self.reduce_linear_bias[label_idx]
                  for label_idx, hx_layer in zip(label_indices, unpacked_hx)]
            hx = torch.stack(hx, axis=0)
            hx = hx.unsqueeze(0)
            lstm_hx = self.nonlinearity(hx)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.BIGRAM:
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            unpacked_hx = []
            for nhx in node_hx:
                # tanh or otherwise limit the size of the output?
                stacked_nhx = self.lstm_input_dropout(torch.cat(nhx, axis=0))
                if stacked_nhx.shape[0] > 1:
                    bigram_hx = torch.cat((stacked_nhx[:-1, :], stacked_nhx[1:, :]), axis=1)
                    bigram_hx = self.reduce_bigram(bigram_hx) / 2
                    stacked_nhx = torch.cat((stacked_nhx, bigram_hx), axis=0)
                unpacked_hx.append(torch.max(stacked_nhx, 0).values)
            packed_hx = torch.stack(unpacked_hx, axis=0).unsqueeze(0)
            hx = self.reduce_linear(packed_hx)
            lstm_hx = self.nonlinearity(hx)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.ATTN:
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            label_hx = [self.constituent_open_embedding(self.constituent_open_tensors[self.constituent_open_map[label]]) for label in labels]
            unpacked_hx = [torch.stack(nhx) for nhx in node_hx]
            unpacked_hx = [torch.cat((lhx.unsqueeze(0).unsqueeze(0), nhx), axis=0) for lhx, nhx in zip(label_hx, unpacked_hx)]
            unpacked_hx = [self.reduce_attn(nhx, nhx, nhx)[0].squeeze(1) for nhx in unpacked_hx]
            unpacked_hx = [self.lstm_input_dropout(torch.max(nhx, 0).values) for nhx in unpacked_hx]
            hx = torch.stack(unpacked_hx, axis=0)
            lstm_hx = self.nonlinearity(hx).unsqueeze(0)
            lstm_cx = None
        elif self.constituency_composition == ConstituencyComposition.KEY or self.constituency_composition == ConstituencyComposition.UNTIED_KEY:
            node_hx = [torch.stack([child.value.tree_hx for child in children]) for children in children_lists]
            # add a position vector to each node_hx
            node_hx = [self.reduce_position(x.reshape(x.shape[0], -1)) for x in node_hx]
            query_hx = [self.reduce_query(nhx) for nhx in node_hx]
            # reshape query for MHA
            query_hx = [nhx.reshape(nhx.shape[0], self.reduce_heads, -1).transpose(0, 1) for nhx in query_hx]
            if self.constituency_composition == ConstituencyComposition.KEY:
                queries = [torch.matmul(nhx, self.reduce_key) for nhx in query_hx]
            else:
                label_indices = [self.constituent_open_map[label] for label in labels]
                queries = [torch.matmul(nhx, self.reduce_key[label_idx]) for nhx, label_idx in zip(query_hx, label_indices)]
            # softmax each head
            weights = [torch.nn.functional.softmax(nhx, dim=1).transpose(1, 2) for nhx in queries]
            value_hx = [self.reduce_value(nhx) for nhx in node_hx]
            value_hx = [nhx.reshape(nhx.shape[0], self.reduce_heads, -1).transpose(0, 1) for nhx in value_hx]
            # use the softmaxes to add up the heads
            unpacked_hx = [torch.matmul(weight, nhx).squeeze(1) for weight, nhx in zip(weights, value_hx)]
            unpacked_hx = [nhx.reshape(-1) for nhx in unpacked_hx]
            hx = torch.stack(unpacked_hx, axis=0).unsqueeze(0)
            lstm_hx = self.nonlinearity(hx)
            lstm_cx = None
        elif self.constituency_composition in (ConstituencyComposition.TREE_LSTM, ConstituencyComposition.TREE_LSTM_CX):
            label_hx = [self.lstm_input_dropout(self.constituent_open_embedding(self.constituent_open_tensors[self.constituent_open_map[label]])) for label in labels]
            label_hx = torch.stack(label_hx).unsqueeze(0)

            max_length = max(len(children) for children in children_lists)

            # stacking will let us do elementwise multiplication faster, hopefully
            node_hx = [[child.value.tree_hx for child in children] for children in children_lists]
            unpacked_hx = [self.lstm_input_dropout(torch.stack(nhx)) for nhx in node_hx]
            unpacked_hx = [nhx.max(dim=0) for nhx in unpacked_hx]
            packed_hx = torch.stack([nhx.values for nhx in unpacked_hx], axis=1)
            #packed_hx = packed_hx.max(dim=0).values

            node_cx = [torch.stack([child.value.tree_cx for child in children]) for children in children_lists]
            node_cx_indices = [uhx.indices.unsqueeze(0) for uhx in unpacked_hx]
            unpacked_cx = [ncx.gather(0, nci).squeeze(0) for ncx, nci in zip(node_cx, node_cx_indices)]
            packed_cx = torch.stack(unpacked_cx, axis=1)

            _, (lstm_hx, lstm_cx) = self.constituent_reduce_lstm(label_hx, (packed_hx, packed_cx))
        else:
            raise ValueError("Unhandled ConstituencyComposition: {}".format(self.constituency_composition))

        constituents = []
        for idx, (label, children) in enumerate(zip(labels, children_lists)):
            children = [child.value.value for child in children]
            if isinstance(label, str):
                node = Tree(label=label, children=children)
            else:
                for value in reversed(label):
                    node = Tree(label=value, children=children)
                    children = node
            constituents.append(Constituent(node, lstm_hx[:, idx, :], lstm_cx[:, idx, :] if lstm_cx is not None else None))
        return constituents

    def push_constituents(self, constituent_stacks, constituents):
        # Another possibility here would be to use output[0, i, :]
        # from the constituency lstm for the value of the new node.
        # This might theoretically make the new constituent include
        # information from neighboring constituents.  However, this
        # lowers the scores of various models.
        # For example, an experiment on ja_alt built this way,
        # averaged over 5 trials, had the following loss in accuracy:
        # 150 epochs: 0.8971 to 0.8953
        # 200 epochs: 0.8985 to 0.8964
        current_nodes = [stack.value for stack in constituent_stacks]

        constituent_input = torch.stack([x.tree_hx[-1:] for x in constituents], axis=1)
        #constituent_input = constituent_input.unsqueeze(0)
        # the constituents are already Constituent(tree, tree_hx, tree_cx)
        return self.constituent_stack.push_states(constituent_stacks, constituents, constituent_input)

    def get_top_constituent(self, constituents):
        """
        Extract only the top constituent from a state's constituent
        sequence, even though it has multiple addition pieces of
        information
        """
        # TreeStack value -> LSTMTreeStack value -> Constituent value -> constituent
        return constituents.value.value.value

    def push_transitions(self, transition_stacks, transitions):
        """
        Push all of the given transitions on to the stack as a batch operations.

        Significantly faster than doing one transition at a time.
        """
        transition_idx = torch.stack([self.transition_tensors[self.transition_map[transition]] for transition in transitions])
        transition_input = self.transition_embedding(transition_idx).unsqueeze(0)
        return self.transition_stack.push_states(transition_stacks, transitions, transition_input)

    def get_top_transition(self, transitions):
        """
        Extract only the top transition from a state's transition
        sequence, even though it has multiple addition pieces of
        information
        """
        # TreeStack value -> LSTMTreeStack value -> transition
        return transitions.value.value

    def forward(self, states):
        """
        Return logits for a prediction of what transition to make next

        We've basically done all the work analyzing the state as
        part of applying the transitions, so this method is very simple

        return shape: (num_states, num_transitions)
        """
        word_hx = torch.stack([state.get_word(state.word_position).hx for state in states])
        transition_hx = torch.stack([self.transition_stack.output(state.transitions) for state in states])
        # this .output() is the output of the constituent stack, not the
        # constituent itself
        # this way, we can, as an option, NOT include the constituents to the left
        # when building the current vector for a constituent
        # and the vector used for inference will still incorporate the entire LSTM
        constituent_hx = torch.stack([self.constituent_stack.output(state.constituents) for state in states])

        hx = torch.cat((word_hx, transition_hx, constituent_hx), axis=1)
        for idx, output_layer in enumerate(self.output_layers):
            hx = self.predict_dropout(hx)
            if not self.maxout_k and idx < len(self.output_layers) - 1:
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
        pred_max = torch.argmax(predictions, dim=1)
        scores = torch.take_along_dim(predictions, pred_max.unsqueeze(1), dim=1)
        pred_max = pred_max.detach().cpu()

        pred_trans = [self.transitions[pred_max[idx]] for idx in range(len(states))]
        if is_legal:
            for idx, (state, trans) in enumerate(zip(states, pred_trans)):
                if not trans.is_legal(state, self):
                    _, indices = predictions[idx, :].sort(descending=True)
                    for index in indices:
                        if self.transitions[index].is_legal(state, self):
                            pred_trans[idx] = self.transitions[index]
                            scores[idx] = predictions[idx, index]
                            break
                    else: # yeah, else on a for loop, deal with it
                        pred_trans[idx] = None
                        scores[idx] = None

        return predictions, pred_trans, scores.squeeze(1)

    def weighted_choice(self, states):
        """
        Generate and return predictions, and randomly choose a prediction weighted by the scores

        TODO: pass in a temperature
        """
        predictions = self.forward(states)
        pred_trans = []
        all_scores = []
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
            all_scores.append(prediction[idx])
        all_scores = torch.stack(all_scores)
        return predictions, pred_trans, all_scores

    def predict_gold(self, states):
        """
        For each State, return the next item in the gold_sequence
        """
        predictions = self.forward(states)
        transitions = [y.gold_sequence[y.num_transitions] for y in states]
        indices = torch.tensor([self.transition_map[t] for t in transitions], device=predictions.device)
        scores = torch.take_along_dim(predictions, indices.unsqueeze(1), dim=1)
        return predictions, transitions, scores.squeeze(1)

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
        config = copy.deepcopy(self.args)
        config['sentence_boundary_vectors'] = config['sentence_boundary_vectors'].name
        config['constituency_composition'] = config['constituency_composition'].name
        config['transition_stack'] = config['transition_stack'].name
        config['constituent_stack'] = config['constituent_stack'].name
        config['transition_scheme'] = config['transition_scheme'].name
        assert isinstance(self.rare_words, set)
        params = {
            'model': model_state,
            'model_type': "LSTM",
            'config': config,
            'transitions': [repr(x) for x in self.transitions],
            'constituents': self.constituents,
            'tags': self.tags,
            'words': self.delta_words,
            'rare_words': list(self.rare_words),
            'root_labels': self.root_labels,
            'constituent_opens': self.constituent_opens,
            'unary_limit': self.unary_limit(),
        }

        return params

