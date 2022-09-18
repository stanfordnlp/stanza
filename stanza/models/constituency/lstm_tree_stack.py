"""
Keeps an LSTM in TreeStack form.

The TreeStack nodes keep the hx and cx for the LSTM, along with a
"value" which represents whatever the user needs to store.

The TreeStacks can be ppped to get back to the previous LSTM state.

The module itself implements two methods: initial_state and push_states
"""

from collections import namedtuple

import torch
import torch.nn as nn

from stanza.models.constituency.tree_stack import TreeStack

Node = namedtuple("Node", ['value', 'lstm_hx', 'lstm_cx'])

class LSTMTreeStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, dropout, uses_boundary_vector, input_dropout):
        super().__init__()

        self.uses_boundary_vector = uses_boundary_vector

        # The start embedding needs to be input_size as we put it through the LSTM
        # A zeros vector needs to be *hidden_size* as we do not put that through the LSTM
        if uses_boundary_vector:
            self.register_parameter('start_embedding', torch.nn.Parameter(0.2 * torch.randn(input_size, requires_grad=True)))
        else:
            self.register_buffer('zeros', torch.zeros(num_lstm_layers, 1, hidden_size))

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers, dropout=dropout)
        self.input_dropout = input_dropout


    def initial_state(self):
        """
        Return an initial state, either based on zeros or based on the initial embedding and LSTM

        Note that LSTM start operation is already batched, in a sense
        The subsequent batch built this way will be used for batch_size trees

        Returns a stack with None value, hx & cx either based on the
        start_embedding or zeros, and no parent.
        """
        if self.uses_boundary_vector:
            start = self.start_embedding.unsqueeze(0).unsqueeze(0)
            output, (hx, cx) = self.lstm(start)
            start = output[0, 0, :]
        else:
            hx = self.zeros
            cx = self.zeros
        return TreeStack(value=Node(None, hx, cx), parent=None, length=1)

    def push_states(self, stacks, values, inputs):
        """
        Starting from a list of current stacks, put the inputs through the LSTM and build new stack nodes.

        N = stacks.len() = values.len()

        inputs must be of shape 1 x N x input_size
        """
        inputs = self.input_dropout(inputs)

        hx = torch.cat([t.value.lstm_hx for t in stacks], axis=1)
        cx = torch.cat([t.value.lstm_cx for t in stacks], axis=1)
        output, (hx, cx) = self.lstm(inputs, (hx, cx))
        new_stacks = [stack.push(Node(transition, hx[:, i:i+1, :], cx[:, i:i+1, :]))
                      for i, (stack, transition) in enumerate(zip(stacks, values))]
        return new_stacks

