"""
A version of the BaseModel which uses LSTMs to predict the correct next transition
based on the current known state.

The primary purpose of this class is to implement the prediction of the next
transition, which is done by concatenating the output of an LSTM operated over
previous transitions, the words, and the partially built constituents.
"""

from collections import namedtuple
from enum import Enum
import logging
from operator import itemgetter
import math
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from stanza.models.common.data import get_long_tensor
from stanza.models.common.utils import unsort
from stanza.models.common.vocab import PAD_ID, UNK_ID
from stanza.models.constituency.base_model import BaseModel
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.tree_stack import TreeStack
from stanza.models.constituency.utils import build_nonlinearity, initialize_linear, TextTooLongError
from stanza.models.constituency.partitioned_transformer import PartitionedTransformerModule

logger = logging.getLogger('stanza')

WordNode = namedtuple("WordNode", ['value', 'hx'])
TransitionNode = namedtuple("TransitionNode", ['value', 'output', 'hx', 'cx'])

# Invariant: the output at the top of the constituency stack will have a
# single dimension
# We do this to maintain consistency between the different operations,
# which sometimes result in different shapes
# This will be unsqueezed in order to put into the next layer if needed
# hx & cx are the hidden & cell states of the LSTM going across constituents
ConstituentNode = namedtuple("ConstituentNode", ['value', 'output', 'hx', 'cx'])
Constituent = namedtuple("Constituent", ['value', 'hx'])

# The sentence boundary vectors are marginally useful at best.
# However, they make it much easier to use non-bert layers as input to
# attention layers, as the attention layers work better when they have
# an index 0 to attend to.
class SentenceBoundary(Enum):
    NONE               = 1
    WORDS              = 2
    EVERYTHING         = 3


class LSTMModel(BaseModel, nn.Module):
    def __init__(self, pretrain, forward_charlm, backward_charlm, bert_model, bert_tokenizer, transitions, constituents, tags, words, rare_words, root_labels, open_nodes, unary_limit, args):
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
        open_nodes: a list of all possible open nodes which will go on the stack
          - this might be different from constituents if there are nodes
            which represent multiple constituents at once
        args: hidden_size, transition_hidden_size, etc as gotten from
          constituency_parser.py

        Note that it might look like a hassle to pass all of this in
        when it can be collected directly from the trees themselves.
        However, that would only work at train time.  At eval or
        pipeline time we will load the lists from the saved model.
        """
        super().__init__()
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
        self.constituent_map = { x: i for (i, x) in enumerate(self.constituents) }
        # precompute tensors for the constituents
        self.register_buffer('constituent_tensors', torch.tensor(range(len(self.constituent_map)), requires_grad=False))

        self.hidden_size = self.args['hidden_size']
        self.transition_hidden_size = self.args['transition_hidden_size']
        self.tag_embedding_dim = self.args['tag_embedding_dim']
        self.transition_embedding_dim = self.args['transition_embedding_dim']
        self.delta_embedding_dim = self.args['delta_embedding_dim']

        self.word_input_size = self.embedding_dim + self.tag_embedding_dim + self.delta_embedding_dim

        if forward_charlm is not None:
            self.add_unsaved_module('forward_charlm', forward_charlm)
            self.add_unsaved_module('forward_charlm_vocab', forward_charlm.char_vocab())
            self.word_input_size += self.forward_charlm.hidden_dim()
        else:
            self.forward_charlm = None
        if backward_charlm is not None:
            self.add_unsaved_module('backward_charlm', backward_charlm)
            self.add_unsaved_module('backward_charlm_vocab', backward_charlm.char_vocab())
            self.word_input_size += self.backward_charlm.hidden_dim()
        else:
            self.backward_charlm = None

        # TODO: add a max_norm?
        self.delta_words = sorted(set(words))
        self.delta_word_map = { word: i+2 for i, word in enumerate(self.delta_words) }
        assert PAD_ID == 0
        assert UNK_ID == 1
        self.delta_embedding = nn.Embedding(num_embeddings = len(self.delta_words)+2,
                                            embedding_dim = self.delta_embedding_dim,
                                            padding_idx = 0)
        self.register_buffer('delta_tensors', torch.tensor(range(len(self.delta_words) + 2), requires_grad=False))

        self.rare_words = set(rare_words)

        self.tags = sorted(list(tags))
        if self.tag_embedding_dim > 0:
            self.tag_map = { t: i+2 for i, t in enumerate(self.tags) }
            self.tag_embedding = nn.Embedding(num_embeddings = len(tags)+2,
                                              embedding_dim = self.tag_embedding_dim,
                                              padding_idx = 0)
            self.register_buffer('tag_tensors', torch.tensor(range(len(self.tags) + 2), requires_grad=False))

        self.transitions = sorted(list(transitions))
        self.transition_map = { t: i for i, t in enumerate(self.transitions) }
        # precompute tensors for the transitions
        self.register_buffer('transition_tensors', torch.tensor(range(len(transitions)), requires_grad=False))
        self.transition_embedding = nn.Embedding(num_embeddings = len(transitions),
                                                 embedding_dim = self.transition_embedding_dim)

        self.num_layers = self.args['num_lstm_layers']
        self.lstm_layer_dropout = self.args['lstm_layer_dropout']

        # also register a buffer of zeros so that we can always get zeros on the appropriate device
        self.register_buffer('word_zeros', torch.zeros(self.hidden_size))
        self.register_buffer('transition_zeros', torch.zeros(self.num_layers, 1, self.transition_hidden_size))
        self.register_buffer('constituent_zeros', torch.zeros(self.num_layers, 1, self.hidden_size))

        # possibly add a couple vectors for bookends of the sentence
        # We put the word_start and word_end here, AFTER counting the
        # charlm dimension, but BEFORE counting the bert dimension,
        # as we want word_start and word_end to not have dimensions
        # for the bert embedding.  The bert model will add its own
        # start and end representation.
        self.sentence_boundary_vectors = self.args['sentence_boundary_vectors']
        if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
            self.register_parameter('word_start', torch.nn.Parameter(torch.randn(self.word_input_size, requires_grad=True)))
            self.register_parameter('word_end', torch.nn.Parameter(torch.randn(self.word_input_size, requires_grad=True)))
            if self.sentence_boundary_vectors is SentenceBoundary.EVERYTHING:
                self.register_parameter('transition_start', torch.nn.Parameter(torch.randn(self.transition_hidden_size, requires_grad=True)))
                self.register_parameter('constituent_start', torch.nn.Parameter(torch.randn(self.hidden_size, requires_grad=True)))

        # we set up the bert AFTER building word_start and word_end
        # so that we can use the charlm endpoint values rather than
        # try to train our own
        if bert_model is not None:
            if bert_tokenizer is None:
                raise ValueError("Cannot have a bert model without a tokenizer")
            self.add_unsaved_module('bert_model', bert_model)
            self.add_unsaved_module('bert_tokenizer', bert_tokenizer)
            self.bert_dim = self.bert_model.config.hidden_size
            self.word_input_size = self.word_input_size + self.bert_dim
            self.is_phobert = self.bert_tokenizer.name_or_path.startswith("vinai/phobert")
        else:
            self.bert_model = None
            self.bert_tokenizer = None
            self.is_phobert = False

        if self.args['pattn_num_heads'] > 0 and self.args['pattn_num_layers'] > 0:
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
        else:
            self.partitioned_transformer_module = None

        self.word_lstm = nn.LSTM(input_size=self.word_input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, dropout=self.lstm_layer_dropout)

        # after putting the word_delta_tag input through the word_lstm, we get back
        # hidden_size * 2 output with the front and back lstms concatenated.
        # this transforms it into hidden_size with the values mixed together
        self.word_to_constituent = nn.Linear(self.hidden_size * 2, self.hidden_size)
        initialize_linear(self.word_to_constituent, self.args['nonlinearity'], self.hidden_size * 2)

        self.transition_lstm = nn.LSTM(input_size=self.transition_embedding_dim, hidden_size=self.transition_hidden_size, num_layers=self.num_layers, dropout=self.lstm_layer_dropout)
        # input_size is hidden_size - could introduce a new constituent_size instead if we liked
        self.constituent_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.lstm_layer_dropout)

        self._transition_scheme = args['transition_scheme']
        if self._transition_scheme is TransitionScheme.TOP_DOWN_UNARY:
            unary_transforms = {}
            for constituent in self.constituent_map:
                unary_transforms[constituent] = nn.Linear(self.hidden_size, self.hidden_size)
            self.unary_transforms = nn.ModuleDict(unary_transforms)

        self.open_nodes = sorted(list(open_nodes))
        # an embedding for the spot on the constituent LSTM taken up by the Open transitions
        # the pattern when condensing constituents is embedding - con1 - con2 - con3 - embedding
        # TODO: try the two ends have different embeddings?
        self.open_node_map = { x: i for (i, x) in enumerate(self.open_nodes) }
        self.open_node_embedding = nn.Embedding(num_embeddings = len(self.open_node_map),
                                                embedding_dim = self.hidden_size)

        if args['combined_dummy_embedding']:
            self.dummy_embedding = self.open_node_embedding
        else:
            self.dummy_embedding = nn.Embedding(num_embeddings = len(self.open_node_map),
                                                embedding_dim = self.hidden_size)
        self.register_buffer('open_node_tensors', torch.tensor(range(len(open_nodes)), requires_grad=False))

        # forward and backward pieces for crunching several
        # constituents into one, combined into a bi-lstm
        # TODO: make the hidden size here an option?
        self.constituent_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, dropout=self.lstm_layer_dropout)
        # affine transformation from bi-lstm reduce to a new hidden layer
        self.reduce_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        initialize_linear(self.reduce_linear, self.args['nonlinearity'], self.hidden_size * 2)

        self.nonlinearity = build_nonlinearity(self.args['nonlinearity'])

        self.word_dropout = nn.Dropout(self.args['word_dropout'])
        self.predict_dropout = nn.Dropout(self.args['predict_dropout'])
        self.lstm_input_dropout = nn.Dropout(self.args['lstm_input_dropout'])

        # matrix for predicting the next transition using word/constituent/transition queues
        # word size + constituency size + transition size
        self.output_layers = self.build_output_layers(self.args['num_output_layers'], len(transitions))

        self.use_constituency_lstm_hx = self.args['constituency_lstm']

        self._unary_limit = unary_limit

    def build_output_layers(self, num_output_layers, final_layer_size):
        """
        Build a ModuleList of Linear transformations for the given num_output_layers

        The final layer size can be specified.
        Initial layer size is the combination of word, constituent, and transition vectors
        Middle layer sizes are self.hidden_size
        """
        middle_layers = num_output_layers - 1
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

    def get_root_labels(self):
        return self.root_labels

    def build_char_representation(self, all_word_labels, device, forward):
        CHARLM_START = "\n"
        CHARLM_END = " "

        if forward:
            charlm = self.forward_charlm
            vocab = self.forward_charlm_vocab
        else:
            charlm = self.backward_charlm
            vocab = self.backward_charlm_vocab

        all_data = []
        for idx, word_labels in enumerate(all_word_labels):
            if not forward:
                word_labels = [x[::-1] for x in reversed(word_labels)]

            chars = [CHARLM_START]
            offsets = []
            for w in word_labels:
                chars.extend(w)
                chars.append(CHARLM_END)
                offsets.append(len(chars) - 1)
            if not forward:
                offsets.reverse()
            chars = vocab.map(chars)
            all_data.append((chars, offsets, len(chars), len(all_data)))

        all_data.sort(key=itemgetter(2), reverse=True)
        chars, char_offsets, char_lens, orig_idx = tuple(zip(*all_data))
        chars = get_long_tensor(chars, len(all_data), pad_id=vocab.unit2id(' ')).to(device=device)

        # TODO: surely this should be stuffed in the charlm model itself rather than done here
        with torch.no_grad():
            output, _, _ = charlm.forward(chars, char_lens)
            res = [output[i, offsets] for i, offsets in enumerate(char_offsets)]
            res = unsort(res, orig_idx)

        return res

    @staticmethod
    def extract_bert_embeddings(tokenizer, model, data, device):
        """
        Extract transformer embeddings using a generic roberta extraction

        data: list of list of string (the text tokens)
        """
        tokenized = tokenizer(data, padding="longest", is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=False)
        list_offsets = [[None] * (len(sentence)+2) for sentence in data]
        for idx in range(len(data)):
            offsets = tokenized.word_ids(batch_index=idx)
            for pos, offset in enumerate(offsets):
                if offset is None:
                    continue
                # this uses the last token piece for any offset by overwriting the previous value
                list_offsets[idx][offset+1] = pos
            list_offsets[idx][0] = 0
            list_offsets[idx][-1] = -1

            if len(offsets) > tokenizer.model_max_length:
                logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(offsets), data[idx])
                raise TextTooLongError(len(offsets), tokenizer.model_max_length, idx, " ".join(data[idx]))

        features = []
        for i in range(int(math.ceil(len(data)/128))):
            with torch.no_grad():
                feature = model(torch.tensor(tokenized['input_ids'][128*i:128*i+128]).to(device), output_hidden_states=True)
                feature = feature[2]
                feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
                features += feature.clone().detach()

        processed = []
        #process the output
        for feature, offsets in zip(features, list_offsets):
            new_sent = feature[offsets]
            processed.append(new_sent)

        return processed

    @staticmethod
    def extract_phobert_embeddings(tokenizer, model, data, device):
        """
        Extract transformer embeddings using a method specifically for phobert

        Since phobert doesn't have the is_split_into_words / tokenized.word_ids(batch_index=0)
        capability, we instead look for @@ to denote a continued token.

        data: list of list of string (the text tokens)
        """
        processed = [] # final product, returns the list of list of word representation
        tokenized_sents = [] # list of sentences, each is a torch tensor with start and end token
        list_tokenized = [] # list of tokenized sentences from phobert
        for idx, sent in enumerate(data):
            #replace \xa0 or whatever the space character is by _ since PhoBERT expects _ between syllables
            tokenized = [word.replace(" ","_") for word in sent]

            #concatenate to a sentence
            sentence = ' '.join(tokenized)

            #tokenize using AutoTokenizer PhoBERT
            tokenized = tokenizer.tokenize(sentence)

            #add tokenized to list_tokenzied for later checking
            list_tokenized.append(tokenized)

            #convert tokens to ids
            sent_ids = tokenizer.convert_tokens_to_ids(tokenized)

            #add start and end tokens to sent_ids
            tokenized_sent = [tokenizer.bos_token_id] + sent_ids + [tokenizer.eos_token_id]

            if len(tokenized_sent) > tokenizer.model_max_length:
                logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(tokenized_sent), data[idx])
                raise TextTooLongError(len(tokenized_sent), tokenizer.model_max_length, idx, " ".join(data[idx]))

            #add to tokenized_sents
            tokenized_sents.append(torch.tensor(tokenized_sent).detach())

            processed_sent = []
            processed.append(processed_sent)

            # done loading bert emb

        size = len(tokenized_sents)

        #padding the inputs
        tokenized_sents_padded = torch.nn.utils.rnn.pad_sequence(tokenized_sents,batch_first=True,padding_value=tokenizer.pad_token_id)

        features = []

        # Feed into PhoBERT 128 at a time in a batch fashion. In testing, the loop was
        # run only 1 time as the batch size seems to be 30
        for i in range(int(math.ceil(size/128))):
            with torch.no_grad():
                feature = model(tokenized_sents_padded[128*i:128*i+128].clone().detach().to(device), output_hidden_states=True)
                # averaging the last four layers worked well for non-VI languages
                feature = feature[2]
                feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
                features += feature.clone().detach()

        assert len(features)==size
        assert len(features)==len(processed)

        #process the output
        #only take the vector of the last word piece of a word/ you can do other methods such as first word piece or averaging.
        # idx2+1 compensates for the start token at the start of a sentence
        # [0] and [-1] grab the start and end representations as well
        offsets = [[0] + [idx2+1 for idx2, _ in enumerate(list_tokenized[idx]) if (idx2 > 0 and not list_tokenized[idx][idx2-1].endswith("@@")) or (idx2==0)] + [-1]
                   for idx, sent in enumerate(processed)]
        processed = [feature[offset] for feature, offset in zip(features, offsets)]

        # This is a list of ltensors
        # Each tensor holds the representation of a sentence extracted from phobert
        return processed

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
            all_forward_chars = self.build_char_representation(all_word_labels, device, forward=True)
            for word_inputs, forward_chars in zip(all_word_inputs, all_forward_chars):
                word_inputs.append(forward_chars)
        if self.backward_charlm is not None:
            all_backward_chars = self.build_char_representation(all_word_labels, device, forward=False)
            for word_inputs, backward_chars in zip(all_word_inputs, all_backward_chars):
                word_inputs.append(backward_chars)

        all_word_inputs = [torch.cat(word_inputs, dim=1) for word_inputs in all_word_inputs]
        if self.sentence_boundary_vectors is not SentenceBoundary.NONE:
            word_start = self.word_start.unsqueeze(0)
            word_end = self.word_end.unsqueeze(0)
            all_word_inputs = [torch.cat([word_start, word_inputs, word_end], dim=0) for word_inputs in all_word_inputs]

        if self.bert_model is not None:
            # BERT embedding extraction
            # result will be len+2 for each sentence
            # we will take 1:-1 if we don't care about the endpoints
            if self.is_phobert:
                bert_embeddings = self.extract_phobert_embeddings(self.bert_tokenizer, self.bert_model, all_word_labels, device)
            else:
                bert_embeddings = self.extract_bert_embeddings(self.bert_tokenizer, self.bert_model, all_word_labels, device)

            if self.sentence_boundary_vectors is SentenceBoundary.NONE:
                bert_embeddings = [be[1:-1] for be in bert_embeddings]
            all_word_inputs = [torch.cat((x, y), axis=1) for x, y in zip(all_word_inputs, bert_embeddings)]

        # Extract partitioned representation
        if self.partitioned_transformer_module is not None:
            partitioned_embeddings = self.partitioned_transformer_module(None, all_word_inputs)
            all_word_inputs = [torch.cat((x, y[:x.shape[0], :]), axis=1) for x, y in zip(all_word_inputs, partitioned_embeddings)]

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
            transition_start = self.transition_start.unsqueeze(0).unsqueeze(0)
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
            constituent_start = self.constituent_start.unsqueeze(0).unsqueeze(0)
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
        return Constituent(value=word, hx=word_node.hx)

    def dummy_constituent(self, dummy):
        label = dummy.label
        open_index = self.open_node_tensors[self.open_node_map[label]]
        hx = self.dummy_embedding(open_index)
        return Constituent(value=dummy, hx=hx)

    def unary_transform(self, constituents, labels):
        top_constituent = constituents.value
        node = top_constituent.value
        hx = top_constituent.output
        for label in reversed(labels):
            node = Tree(label=label, children=[node])
            hx = self.unary_transforms[label](hx)
            # non-linearity after the unary transform
            hx = self.nonlinearity(hx)
        top_constituent = Constituent(value=node, hx=hx)
        return top_constituent

    def build_constituents(self, labels, children_lists):
        """
        Build new constituents with the given label from the list of children

        labels is a list of labels for each of the new nodes to construct
        children_lists is a list of children that go under each of the new nodes
        lists of each are used so that we can stack operations
        """
        label_hx = [self.open_node_embedding(self.open_node_tensors[self.open_node_map[label]]) for label in labels]

        max_length = max(len(children) for children in children_lists)
        zeros = torch.zeros(self.hidden_size, device=label_hx[0].device)
        node_hx = [[child.output for child in children] for children in children_lists]
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
        lstm_output = lstm_output[1][0]
        forward_hx = lstm_output[-2, :]
        backward_hx = lstm_output[-1, :]

        hx = self.reduce_linear(torch.cat((forward_hx, backward_hx), axis=1))
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
            constituents.append(Constituent(value=node, hx=hx[idx, :]))
        return constituents

    def push_constituents(self, constituent_stacks, constituents):
        current_nodes = [stack.value for stack in constituent_stacks]

        constituent_input = torch.stack([x.hx for x in constituents])
        constituent_input = constituent_input.unsqueeze(0)
        constituent_input = self.lstm_input_dropout(constituent_input)

        hx = torch.cat([current_node.hx for current_node in current_nodes], axis=1)
        cx = torch.cat([current_node.cx for current_node in current_nodes], axis=1)
        output, (hx, cx) = self.constituent_lstm(constituent_input, (hx, cx))
        if self.use_constituency_lstm_hx:
            new_stacks = [stack.push(ConstituentNode(constituent.value, output[0, i, :], hx[:, i:i+1, :], cx[:, i:i+1, :]))
                          for i, (stack, constituent) in enumerate(zip(constituent_stacks, constituents))]
        else:
            new_stacks = [stack.push(ConstituentNode(constituent.value, constituents[i].hx, hx[:, i:i+1, :], cx[:, i:i+1, :]))
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

        hx = torch.cat([t.value.hx for t in transition_stacks], axis=1)
        cx = torch.cat([t.value.cx for t in transition_stacks], axis=1)
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

    def unary_limit(self):
        return self._unary_limit

    def transition_scheme(self):
        return self._transition_scheme

    def has_unary_transitions(self):
        return self._transition_scheme is TransitionScheme.TOP_DOWN_UNARY

    def is_top_down(self):
        return self._transition_scheme in (TransitionScheme.TOP_DOWN, TransitionScheme.TOP_DOWN_UNARY, TransitionScheme.TOP_DOWN_COMPOUND)

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
        constituent_hx = torch.stack([state.constituents.value.hx[-1, 0, :] for state in states])

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
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.unsaved_modules]
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
            'open_nodes': self.open_nodes,
        }

        return params

