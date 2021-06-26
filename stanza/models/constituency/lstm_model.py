from collections import namedtuple
import torch
import torch.nn as nn

from stanza.models.common.vocab import PAD_ID, UNK_ID
from stanza.models.constituency.base_model import BaseModel

from stanza.models.constituency.parse_tree import Tree

WordNode = namedtuple("WordNode", ['value', 'hx', 'cx'])
TransitionNode = namedtuple("TransitionNode", ['value', 'hx', 'cx'])
# Invariant: the hx at the top of the constituency stack will have a
# single dimension
# We do this to maintain consistency between the different operations,
# which sometimes result in different shapes
# This will be unsqueezed in order to put into the next layer if needed
# TODO: this is used for the nodes before they get pushed as well
# perhaps need to separate those uses into different things
ConstituentNode = namedtuple("ConstituentNode", ['value', 'hx', 'cx'])

class LSTMModel(BaseModel, nn.Module):
    """
    Run an LSTM over each item as we put it in the queue

    args:
      hidden_size
      transition_embedding_dim
      constituent_embedding_dim
    """
    def __init__(self, pretrain, transitions, constituents, tags, root_labels, args):
        """
        constituents: a list of all possible constituents in the treebank
        tags: a list of all possible tags in the treebank

        Note that it might look like a hassle to pass all of this in
        when it can be collected directly from the trees themselves.
        However, that would only work at train time.  At eval or
        pipeline time we will load the lists from the saved model.
        """
        super().__init__()
        self.unsaved_modules = []

        emb_matrix = pretrain.emb
        self.add_unsaved_module('embedding', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
        self.vocab_map = { word: i for i, word in enumerate(pretrain.vocab) }
        # precompute tensors for the word indices
        # the tensors should be put on the GPU if needed with a call to cuda()
        self.register_buffer('vocab_tensors', torch.tensor(range(len(pretrain.vocab)), requires_grad=False))
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        self.tags = sorted(list(tags))
        self.root_labels = sorted(list(root_labels))
        constituents = sorted(list(constituents))
        self.constituents = { x: i for (i, x) in enumerate(constituents) }
        # precompute tensors for the constituents
        self.register_buffer('constituent_tensors', torch.tensor(range(len(constituents)), requires_grad=False))

        # TODO: add a tag embedding and a delta embedding
        self.word_input_size = self.embedding_dim
        self.hidden_size = args['hidden_size']
        self.transition_embedding_dim = args['transition_embedding_dim']

        self.transitions = sorted(transitions)
        self.transition_map = { t: i for i, t in enumerate(self.transitions) }
        # precompute tensors for the transitions
        self.register_buffer('transition_tensors', torch.tensor(range(len(transitions)), requires_grad=False))
        # TODO: include max_norm?
        self.transition_embedding = nn.Embedding(num_embeddings = len(transitions),
                                                 embedding_dim = self.transition_embedding_dim)

        # also register a buffer of zeros so that we can always get zeros on the appropriate device
        self.register_buffer('zeros', torch.zeros(self.hidden_size))

        self.word_lstm = nn.LSTMCell(input_size=self.word_input_size, hidden_size=self.hidden_size)
        self.transition_lstm = nn.LSTMCell(input_size=self.transition_embedding_dim, hidden_size=self.hidden_size)
        # input_size is hidden_size - could introduce a new constituent_size instead if we liked
        self.constituent_lstm = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)

        # when pushing a new constituent made from a single word_tag pair
        # note that the word_tag pair has been mapped to hidden_size at this point
        # TODO: test if that is best
        self.word_to_constituent = nn.Linear(self.hidden_size, self.hidden_size)

        unary_transforms = {}
        for constituent in self.constituents:
            unary_transforms[constituent] = nn.Linear(self.hidden_size, self.hidden_size)
        self.unary_transforms = nn.ModuleDict(unary_transforms)

        self.dummy_embedding = nn.Embedding(num_embeddings = len(self.constituents),
                                            embedding_dim = self.hidden_size)

        # TODO: the original paper suggests a BI-LSTM.  This is just a single direction LSTM
        # the original paper also includes an initial input to the LSTM which says what
        # constituent is being reduced
        # TODO: also try with making the LSTM "semantically untied"
        self.reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size)

        # matrix for predicting the next transition using word/constituent/transition queues
        self.W = nn.Linear(self.hidden_size * 3, len(transitions))

    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)


    def push_word(self, word_queue, word):
        """
        word actually means a ParseTree with a tag node and word node
        """
        # TODO: here we can append a tag embedding as well
        word_idx = self.vocab_map.get(word.children[0].label, UNK_ID)
        word_idx = self.vocab_tensors[word_idx]
        word_input = self.embedding(word_idx)
        word_input = word_input.unsqueeze(0)

        current_node = word_queue.value
        if current_node:
            cx = current_node.cx
            hx = current_node.hx
            hx, cx = self.word_lstm(word_input, (hx, cx))
        else:
            hx, cx = self.word_lstm(word_input)
        return word_queue.push(WordNode(word, hx, cx))

    def get_top_word(self, word_queue):
        word_node = word_queue.value
        return word_node.value

    def transform_word_to_constituent(self, state):
        word_node = state.word_queue.value
        word = word_node.value
        hx = word_node.hx.squeeze()
        hx = self.word_to_constituent(hx)
        return ConstituentNode(value=word, hx=hx, cx=None)

    def dummy_constituent(self, dummy):
        label = dummy.label
        constituent_index = self.constituent_tensors[self.constituents[label]]
        hx = self.dummy_embedding(constituent_index)
        return ConstituentNode(value=dummy, hx=hx, cx=None)

    def unary_transform(self, constituents, labels):
        top_constituent = constituents.value
        for label in reversed(labels):
            node = top_constituent.value
            node = Tree(label=label, children=[node])
            hx = top_constituent.hx
            hx = self.unary_transforms[label](hx)
            top_constituent = ConstituentNode(value=node, hx=hx, cx=None)
        return top_constituent

    def build_constituent(self, label, children):
        hx = [child.hx for child in children]
        hx = torch.stack(hx)
        hx = hx.unsqueeze(1)
        # should now be: (#nodes, 1, hidden_dim)
        # transform...
        hx = self.reduce_lstm(hx)[0]
        # take just the output of the final layer
        hx = hx[-1, 0, :]

        node = Tree(label=label, children=[child.value for child in children])
        return ConstituentNode(value=node, hx=hx, cx=None)

    def push_constituent(self, constituents, constituent):
        current_node = constituents.value

        # TODO: make the dimensions on the stack consistent
        # shift & unary make different dimension results
        constituent_input = constituent.hx
        constituent_input = constituent_input.unsqueeze(0)

        if current_node:
            hx = current_node.hx.unsqueeze(0)
            cx = current_node.cx
            hx, cx = self.constituent_lstm(constituent_input, (hx, cx))
        else:
            hx, cx = self.constituent_lstm(constituent_input)
        hx = hx.squeeze(0)
        new_node = ConstituentNode(constituent.value, hx, cx)
        return constituents.push(new_node)

    def get_top_constituent(self, constituents):
        constituent_node = constituents.value
        return constituent_node.value

    def push_transition(self, transitions, transition):
        transition_idx = self.transition_tensors[self.transition_map[transition]]
        transition_input = self.transition_embedding(transition_idx)
        transition_input = transition_input.unsqueeze(0)

        current_node = transitions.value
        if current_node:
            cx = current_node.cx
            hx = current_node.hx
            hx, cx = self.transition_lstm(transition_input, (hx, cx))
        else:
            hx, cx = self.transition_lstm(transition_input)
        return transitions.push(TransitionNode(transition, hx, cx))

    def get_top_transition(self, transitions):
        transition_node = transitions.value
        return transition_node.value

    def forward(self, state):
        """
        Return logits for a prediction of what transition to make next

        We've basically done all the work analyzing the state as
        part of applying the transitions, so this method is very simple
        """
        # TODO: could make the sentinel have hx,cx=0
        # would simplify the earlier code blocks
        # TODO: make the word_hx always dim 1?
        word_hx = state.word_queue.value
        word_hx = word_hx.hx.squeeze(0) if word_hx else self.zeros

        # TODO: also, ensure that transition_hx is always dim 1
        transition_hx = state.transitions.value
        transition_hx = transition_hx.hx if transition_hx else self.zeros
        if len(transition_hx.shape) == 2:
            transition_hx = transition_hx.squeeze(0)

        constituent_hx = state.constituents.value
        constituent_hx = constituent_hx.hx if constituent_hx else self.zeros
        hx = torch.cat((word_hx, transition_hx, constituent_hx))
        return self.W(hx)

    # TODO: merge this with forward?
    def predict(self, state, is_legal=False):
        predictions = self.forward(state)
        pred_max = torch.argmax(predictions).item()
        trans = self.transitions[pred_max]
        if not is_legal or trans.is_legal(state, self):
            return predictions, trans
        _, indices = predictions.sort(descending=True)
        for index in indices:
            if self.transitions[index].is_legal(state, self):
                return predictions, self.transitions[index]

        return predictions, None
