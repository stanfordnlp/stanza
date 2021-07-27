from collections import namedtuple
import logging
import torch
import torch.nn as nn

from stanza.models.common.vocab import PAD_ID, UNK_ID
from stanza.models.constituency.base_model import BaseModel
from stanza.models.constituency.tree_stack import TreeStack

from stanza.models.constituency.parse_tree import Tree

logger = logging.getLogger('stanza')

WordNode = namedtuple("WordNode", ['value', 'hx'])
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

        # TODO: add a delta embedding
        self.hidden_size = self.args['hidden_size']
        self.tag_embedding_dim = self.args['tag_embedding_dim']
        self.transition_embedding_dim = self.args['transition_embedding_dim']
        self.word_input_size = self.embedding_dim + self.tag_embedding_dim

        self.tags = sorted(list(tags))
        self.tag_map = { t: i for i, t in enumerate(self.tags) }
        self.tag_embedding = nn.Embedding(num_embeddings = len(tags),
                                          embedding_dim = self.tag_embedding_dim)
        self.register_buffer('tag_tensors', torch.tensor(range(len(self.tags)), requires_grad=False))

        self.transitions = sorted(transitions)
        self.transition_map = { t: i for i, t in enumerate(self.transitions) }
        # precompute tensors for the transitions
        self.register_buffer('transition_tensors', torch.tensor(range(len(transitions)), requires_grad=False))
        # TODO: include max_norm?
        self.transition_embedding = nn.Embedding(num_embeddings = len(transitions),
                                                 embedding_dim = self.transition_embedding_dim)

        # also register a buffer of zeros so that we can always get zeros on the appropriate device
        self.register_buffer('zeros', torch.zeros(self.hidden_size))

        self.word_lstm = nn.LSTM(input_size=self.word_input_size, hidden_size=self.hidden_size)
        self.transition_lstm = nn.LSTMCell(input_size=self.transition_embedding_dim, hidden_size=self.hidden_size)
        # input_size is hidden_size - could introduce a new constituent_size instead if we liked
        self.constituent_lstm = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)

        # when pushing a new constituent made from a single word_tag pair
        # note that the word_tag pair has been mapped to hidden_size at this point
        # TODO: test if that is best
        self.word_to_constituent = nn.Linear(self.hidden_size, self.hidden_size)

        unary_transforms = {}
        for constituent in self.constituent_map:
            unary_transforms[constituent] = nn.Linear(self.hidden_size, self.hidden_size)
        self.unary_transforms = nn.ModuleDict(unary_transforms)

        # an embedding for the spot on the constituent LSTM taken up by the Open transitions
        self.dummy_embedding = nn.Embedding(num_embeddings = len(self.constituent_map),
                                            embedding_dim = self.hidden_size)

        # an embedding for the first symbol in the reduce lstm
        # TODO: try both directions have different embeddings?
        self.constituent_embedding = nn.Embedding(num_embeddings = len(self.constituent_map),
                                                  embedding_dim = self.hidden_size)

        # forward and backward pieces to make a bi-lstm
        # TODO: make the hidden size here an option?
        self.forward_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.backward_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size)
        # affine transformation from bi-lstm reduce to a new hidden layer
        self.reduce_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.tanh = nn.Tanh()
        self.word_dropout = nn.Dropout(self.args['word_dropout'])

        # matrix for predicting the next transition using word/constituent/transition queues
        self.W = nn.Linear(self.hidden_size * 3, len(transitions))

    def add_unsaved_module(self, name, module):
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def get_root_labels(self):
        return self.root_labels

    def initial_word_queue(self, tagged_words):
        word_idx = torch.stack([self.vocab_tensors[self.vocab_map.get(word.children[0].label, UNK_ID)] for word in tagged_words])
        word_input = self.embedding(word_idx)

        try:
            tag_idx = torch.stack([self.tag_tensors[self.tag_map[word.label]] for word in tagged_words])
            tag_input = self.tag_embedding(tag_idx)
        except KeyError as e:
            raise KeyError("Constituency parser not trained with tag {}".format(str(e))) from e

        # now of size sentence x input
        word_input = torch.cat([word_input, tag_input], dim=1)
        # now of size sentence x 1 x input
        word_input = word_input.unsqueeze(1)
        word_input = self.word_dropout(word_input)
        outputs, _ = self.word_lstm(word_input)

        word_queue = TreeStack(value=WordNode(None, self.zeros))
        for idx, tag_node in enumerate(tagged_words):
            word_queue = word_queue.push(WordNode(tag_node, outputs[idx, 0, :].squeeze()))
        return word_queue

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
        constituent_index = self.constituent_tensors[self.constituent_map[label]]
        hx = self.dummy_embedding(constituent_index)
        return ConstituentNode(value=dummy, hx=hx, cx=None)

    def unary_transform(self, constituents, labels):
        top_constituent = constituents.value
        for label in reversed(labels):
            node = top_constituent.value
            node = Tree(label=label, children=[node])
            hx = top_constituent.hx
            hx = self.unary_transforms[label](hx)
            # non-linearity after the unary transform
            hx = self.tanh(hx)
            top_constituent = ConstituentNode(value=node, hx=hx, cx=None)
        return top_constituent

    def build_constituent(self, label, children):
        constituent_index = self.constituent_tensors[self.constituent_map[label]]
        node_hx = [child.hx for child in children]
        label_hx = [self.constituent_embedding(constituent_index)]

        forward_hx = torch.stack(label_hx + node_hx)
        forward_hx = forward_hx.unsqueeze(1)
        # should now be: (#nodes, 1, hidden_dim)
        # transform...
        forward_hx = self.forward_reduce_lstm(forward_hx)[0]
        # take just the output of the final layer
        forward_hx = forward_hx[-1, 0, :]

        node_hx.reverse()
        backward_hx = torch.stack(label_hx + node_hx)
        backward_hx = backward_hx.unsqueeze(1)
        # should now be: (#nodes, 1, hidden_dim)
        # transform...
        backward_hx = self.backward_reduce_lstm(backward_hx)[0]
        # take just the output of the final layer
        backward_hx = backward_hx[-1, 0, :]

        hx = self.reduce_linear(torch.cat((forward_hx, backward_hx)))
        # TODO: try others, like relu, to see if they also do the job but faster
        hx = self.tanh(hx)

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
        word_hx = state.word_queue.value.hx

        # TODO: ensure that transition_hx is always dim 1
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
        'root_labels': model.root_labels,
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
                          root_labels=checkpoint['root_labels'],
                          args=checkpoint['config'])
    else:
        raise ValueError("Unknown model type {}".format(model_type))
    model.load_state_dict(checkpoint['model'], strict=False)

    logger.debug("-- MODEL CONFIG --")
    for k in model.args.keys():
        logger.debug("  --{}: {}".format(k, model.args[k]))

    return model

