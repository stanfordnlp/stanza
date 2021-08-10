from collections import namedtuple
import logging
import random
import torch
import torch.nn as nn

from stanza.models.common.vocab import PAD_ID, UNK_ID
from stanza.models.constituency.base_model import BaseModel
from stanza.models.constituency.tree_stack import TreeStack

from stanza.models.constituency.parse_tree import Tree

logger = logging.getLogger('stanza')

WordNode = namedtuple("WordNode", ['value', 'hx'])
TransitionNode = namedtuple("TransitionNode", ['value', 'output', 'hx', 'cx'])

# Invariant: the hx at the top of the constituency stack will have a
# single dimension
# We do this to maintain consistency between the different operations,
# which sometimes result in different shapes
# This will be unsqueezed in order to put into the next layer if needed
ConstituentNode = namedtuple("ConstituentNode", ['value', 'output', 'hx', 'cx'])
Constituent = namedtuple("Constituent", ['value', 'hx'])


class LSTMModel(BaseModel, nn.Module):
    """
    Run an LSTM over each item as we put it in the queue

    args:
      hidden_size
      transition_embedding_dim
      constituent_embedding_dim
    """
    def __init__(self, pretrain, transitions, constituents, tags, words, rare_words, root_labels, open_nodes, args):
        """
        constituents: a list of all possible constituents in the treebank
        tags: a list of all possible tags in the treebank
        open_nodes: a list of all possible open nodes which will go on the stack
          - this might be different from constituents if there are nodes
            which represent multiple constituents at once

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
        self.vocab_map = { word: i for i, word in enumerate(pretrain.vocab) }
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

        # TODO: add a max_norm?
        self.delta_words = sorted(list(words))
        self.delta_word_map = { word: i+2 for i, word in enumerate(self.delta_words) }
        assert PAD_ID == 0
        assert UNK_ID == 1
        self.delta_embedding = nn.Embedding(num_embeddings = len(self.delta_words)+2,
                                            embedding_dim = self.delta_embedding_dim,
                                            padding_idx = 0)
        self.register_buffer('delta_tensors', torch.tensor(range(len(self.delta_words) + 2), requires_grad=False))

        self.rare_words = set(rare_words)

        self.tags = sorted(list(tags))
        self.tag_map = { t: i for i, t in enumerate(self.tags) }
        self.tag_embedding = nn.Embedding(num_embeddings = len(tags),
                                          embedding_dim = self.tag_embedding_dim)
        self.register_buffer('tag_tensors', torch.tensor(range(len(self.tags)), requires_grad=False))

        self.transitions = sorted(list(transitions))
        self.transition_map = { t: i for i, t in enumerate(self.transitions) }
        # precompute tensors for the transitions
        self.register_buffer('transition_tensors', torch.tensor(range(len(transitions)), requires_grad=False))
        # TODO: include max_norm?
        self.transition_embedding = nn.Embedding(num_embeddings = len(transitions),
                                                 embedding_dim = self.transition_embedding_dim)

        self.num_layers = args['num_lstm_layers']

        # also register a buffer of zeros so that we can always get zeros on the appropriate device
        self.register_buffer('zeros', torch.zeros(self.hidden_size))
        self.register_buffer('transition_zeros', torch.zeros(self.num_layers, 1, self.transition_hidden_size))
        self.register_buffer('constituent_zeros', torch.zeros(self.num_layers, 1, self.hidden_size))

        self.word_lstm = nn.LSTM(input_size=self.word_input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.transition_lstm = nn.LSTM(input_size=self.transition_embedding_dim, hidden_size=self.transition_hidden_size, num_layers=self.num_layers)
        # input_size is hidden_size - could introduce a new constituent_size instead if we liked
        self.constituent_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        # when pushing a new constituent made from a single word_tag pair
        # note that the word_tag pair has been mapped to hidden_size at this point
        # also including word_tag pair - could try more configuratioins and sizes
        self.word_to_constituent = nn.Linear(self.hidden_size + self.word_input_size, self.hidden_size)

        self.use_compound_unary = args['use_compound_unary']
        if self.use_compound_unary:
            unary_transforms = {}
            for constituent in self.constituent_map:
                unary_transforms[constituent] = nn.Linear(self.hidden_size, self.hidden_size)
            self.unary_transforms = nn.ModuleDict(unary_transforms)

        self.open_nodes = sorted(list(open_nodes))
        # an embedding for the spot on the constituent LSTM taken up by the Open transitions
        # TODO: try both directions have different embeddings?
        self.open_node_map = { x: i for (i, x) in enumerate(self.open_nodes) }
        self.open_node_embedding = nn.Embedding(num_embeddings = len(self.open_node_map),
                                                embedding_dim = self.hidden_size)
        self.dummy_embedding = nn.Embedding(num_embeddings = len(self.open_node_map),
                                            embedding_dim = self.hidden_size)
        self.register_buffer('open_node_tensors', torch.tensor(range(len(open_nodes)), requires_grad=False))

        # forward and backward pieces to make a bi-lstm
        # TODO: make the hidden size here an option?
        self.forward_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.backward_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        # affine transformation from bi-lstm reduce to a new hidden layer
        self.reduce_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.args['nonlinearity'] == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif self.args['nonlinearity'] == 'relu':
            self.nonlinearity = nn.ReLU()
        else:
            raise ValueError('Chosen value of nonlinearity, "%s", not handled' % self.args['nonlinearity'])

        self.word_dropout = nn.Dropout(self.args['word_dropout'])
        self.predict_dropout = nn.Dropout(self.args['predict_dropout'])

        # matrix for predicting the next transition using word/constituent/transition queues
        # word size + constituency size + transition size
        self.W = nn.Linear(self.hidden_size * 2 + self.transition_hidden_size, len(transitions))

    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def get_root_labels(self):
        return self.root_labels

    def initial_word_queue(self, tagged_words):
        word_idx = torch.stack([self.vocab_tensors[self.vocab_map.get(word.children[0].label, UNK_ID)] for word in tagged_words])
        word_input = self.embedding(word_idx)

        # this occasionally learns UNK at train time
        word_labels = [word.children[0].label for word in tagged_words]
        if self.training:
            for idx, word in enumerate(word_labels):
                if word in self.rare_words and random.random() < self.args['rare_word_unknown_frequency']:
                    word_labels[idx] = None
        delta_idx = torch.stack([self.delta_tensors[self.delta_word_map.get(word, UNK_ID)] for word in word_labels])

        delta_input = self.delta_embedding(delta_idx)

        try:
            tag_idx = torch.stack([self.tag_tensors[self.tag_map[word.label]] for word in tagged_words])
            tag_input = self.tag_embedding(tag_idx)
        except KeyError as e:
            raise KeyError("Constituency parser not trained with tag {}".format(str(e))) from e

        # now of size sentence x input
        word_input = torch.cat([word_input, delta_input, tag_input], dim=1)
        # now of size sentence x 1 x input
        word_input = word_input.unsqueeze(1)
        word_input = self.word_dropout(word_input)
        outputs, _ = self.word_lstm(word_input)
        outputs = torch.cat((outputs, word_input), axis=2)
        outputs = self.word_to_constituent(outputs)

        word_queue = TreeStack(value=WordNode(None, self.zeros))
        for idx, tag_node in enumerate(tagged_words):
            word_queue = word_queue.push(WordNode(tag_node, outputs[idx, 0, :].squeeze()))
        return word_queue

    def initial_transitions(self):
        return TreeStack(value=TransitionNode(None, self.transition_zeros[-1, 0, :], self.transition_zeros, self.transition_zeros))

    def initial_constituents(self):
        return TreeStack(value=ConstituentNode(None, self.constituent_zeros[-1, 0, :], self.constituent_zeros, self.constituent_zeros))

    def get_top_word(self, word_queue):
        word_node = word_queue.value
        return word_node.value

    def transform_word_to_constituent(self, state):
        word_node = state.word_queue.value
        word = word_node.value
        hx = word_node.hx
        return Constituent(value=word, hx=hx)

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

    def build_constituent(self, label, children):
        open_index = self.open_node_tensors[self.open_node_map[label]]
        node_hx = [child.output for child in children]
        label_hx = [self.open_node_embedding(open_index)]

        forward_hx = torch.stack(label_hx + node_hx, axis=0)
        # should now be: (#nodes, 1, hidden_dim)
        forward_hx = forward_hx.unsqueeze(1)
        # transform...
        forward_hx = self.forward_reduce_lstm(forward_hx)[0]
        # take just the output of the final layer
        forward_hx = forward_hx[-1, 0, :]

        node_hx.reverse()
        backward_hx = torch.stack(label_hx + node_hx, axis=0)
        backward_hx = backward_hx.unsqueeze(1)
        # should now be: (#nodes, 1, hidden_dim)
        # transform...
        backward_hx = self.backward_reduce_lstm(backward_hx)[0]
        # take just the output of the final layer
        backward_hx = backward_hx[-1, 0, :]

        hx = self.reduce_linear(torch.cat((forward_hx, backward_hx)))
        hx = self.nonlinearity(hx)

        children = [child.value for child in children]
        if isinstance(label, str):
            node = Tree(label=label, children=children)
        else:
            for value in reversed(label):
                node = Tree(label=value, children=children)
                children = node
        return Constituent(value=node, hx=hx)

    def push_constituent(self, constituents, constituent):
        current_node = constituents.value

        constituent_input = constituent.hx
        constituent_input = constituent_input.unsqueeze(0).unsqueeze(0)

        hx = current_node.hx
        cx = current_node.cx
        output, (hx, cx) = self.constituent_lstm(constituent_input, (hx, cx))
        new_node = ConstituentNode(constituent.value, output.squeeze(), hx, cx)
        return constituents.push(new_node)

    def push_constituents(self, constituent_stacks, constituents):
        current_nodes = [stack.value for stack in constituent_stacks]

        constituent_input = torch.stack([x.hx for x in constituents])
        constituent_input = constituent_input.unsqueeze(0)

        hx = torch.cat([current_node.hx for current_node in current_nodes], axis=1)
        cx = torch.cat([current_node.cx for current_node in current_nodes], axis=1)
        output, (hx, cx) = self.constituent_lstm(constituent_input, (hx, cx))
        new_stacks = [stack.push(ConstituentNode(constituent.value, output[0, i, :], hx[:, i:i+1, :], cx[:, i:i+1, :]))
                      for i, (stack, constituent) in enumerate(zip(constituent_stacks, constituents))]
        return new_stacks

    def get_top_constituent(self, constituents):
        constituent_node = constituents.value
        return constituent_node.value

    def push_transition(self, transitions, transition):
        transition_idx = self.transition_tensors[self.transition_map[transition]]
        transition_input = self.transition_embedding(transition_idx)
        transition_input = transition_input.unsqueeze(0).unsqueeze(0)

        current_node = transitions.value
        cx = current_node.cx
        hx = current_node.hx
        output, (hx, cx) = self.transition_lstm(transition_input, (hx, cx))
        return transitions.push(TransitionNode(transition, output.squeeze(), hx, cx))

    def push_transitions(self, transition_stacks, transitions):
        transition_idx = torch.stack([self.transition_tensors[self.transition_map[transition]] for transition in transitions])
        transition_input = self.transition_embedding(transition_idx).unsqueeze(0)

        hx = torch.cat([t.value.hx for t in transition_stacks], axis=1)
        cx = torch.cat([t.value.cx for t in transition_stacks], axis=1)
        output, (hx, cx) = self.transition_lstm(transition_input, (hx, cx))
        new_stacks = [stack.push(TransitionNode(transition, output[0, i, :], hx[:, i:i+1, :], cx[:, i:i+1, :]))
                      for i, (stack, transition) in enumerate(zip(transition_stacks, transitions))]
        return new_stacks

    def get_top_transition(self, transitions):
        transition_node = transitions.value
        return transition_node.value

    def has_unary_transitions(self):
        return self.use_compound_unary

    def forward(self, states):
        """
        Return logits for a prediction of what transition to make next

        We've basically done all the work analyzing the state as
        part of applying the transitions, so this method is very simple
        """
        word_hx = torch.stack([state.word_queue.value.hx for state in states])
        transition_hx = torch.stack([state.transitions.value.output for state in states])
        constituent_hx = torch.stack([state.constituents.value.output for state in states])

        hx = torch.cat((word_hx, transition_hx, constituent_hx), axis=1)
        hx = self.predict_dropout(hx)
        return self.W(hx)

    # TODO: merge this with forward?
    def predict(self, states, is_legal=False):
        predictions = self.forward(states)
        pred_max = torch.argmax(predictions, axis=1)

        pred_trans = [None] * len(states)
        for idx, state in enumerate(states):
            trans = self.transitions[pred_max[idx]]
            if not is_legal or trans.is_legal(state, self):
                pred_trans[idx] = trans
            else:
                _, indices = predictions[idx, :].sort(descending=True)
                for index in indices:
                    if self.transitions[index].is_legal(state, self):
                        pred_trans[idx] = self.transitions[index]
                        break

        return predictions, pred_trans


def save(filename, model, skip_modules=True):
    model_state = model.state_dict()
    # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
    if skip_modules:
        skipped = [k for k in model_state.keys() if k.split('.')[0] in model.unsaved_modules]
        for k in skipped:
            del model_state[k]
    params = {
        'model': model_state,
        'model_type': "LSTM",
        'config': model.args,
        'transitions': model.transitions,
        'constituents': model.constituents,
        'tags': model.tags,
        'words': model.delta_words,
        'rare_words': model.rare_words,
        'root_labels': model.root_labels,
        'open_nodes': model.open_nodes,
    }

    torch.save(params, filename, _use_new_zipfile_serialization=False)
    logger.info("Model saved to {}".format(filename))


def load(filename, pretrain):
    try:
        checkpoint = torch.load(filename, lambda storage, loc: storage)
    except BaseException:
        logger.exception("Cannot load model from {}".format(filename))
        raise
    logger.debug("Loaded model {}".format(filename))

    model_type = checkpoint['model_type']
    if model_type == 'LSTM':
        model = LSTMModel(pretrain=pretrain,
                          transitions=checkpoint['transitions'],
                          constituents=checkpoint['constituents'],
                          tags=checkpoint['tags'],
                          words=checkpoint['words'],
                          rare_words=checkpoint['rare_words'],
                          root_labels=checkpoint['root_labels'],
                          open_nodes=checkpoint['open_nodes'],
                          args=checkpoint['config'])
    else:
        raise ValueError("Unknown model type {}".format(model_type))
    model.load_state_dict(checkpoint['model'], strict=False)

    logger.debug("-- MODEL CONFIG --")
    for k in model.args.keys():
        logger.debug("  --{}: {}".format(k, model.args[k]))

    return model

