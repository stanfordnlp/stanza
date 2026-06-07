"""
Keeps an LSTM in TreeStack form.

The TreeStack nodes keep the hx and cx for the LSTM, along with a
"value" which represents whatever the user needs to store.

The TreeStacks can be ppped to get back to the previous LSTM state.

The module itself implements three methods: initial_state, push_states, output
"""

from collections import namedtuple

import torch
import torch.nn as nn

from stanza.models.common.utils import initialize_forget_gate_bias
from stanza.models.constituency.tree_stack import TreeStack

Node = namedtuple("Node", ['value', 'lstm_hx', 'lstm_cx', 'output'])

class LSTMTreeStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, dropout, uses_boundary_vector, input_dropout, forget_bias):
        """
        Prepare LSTM and parameters

        input_size: dimension of the inputs to the LSTM
        hidden_size: LSTM internal & output dimension
        num_lstm_layers: how many layers of LSTM to use
        dropout: value of the LSTM dropout
        uses_boundary_vector: if set, learn a start_embedding parameter.  otherwise, use zeros
        input_dropout: an nn.Module to dropout inputs.  TODO: allow a float parameter as well
        """
        super().__init__()

        self.uses_boundary_vector = uses_boundary_vector

        # The start embedding needs to be input_size as we put it through the LSTM
        if uses_boundary_vector:
            self.register_parameter('start_embedding', torch.nn.Parameter(0.2 * torch.randn(input_size, requires_grad=True)))
        else:
            self.register_buffer('input_zeros',  torch.zeros(num_lstm_layers, input_size))
            self.register_buffer('hidden_zeros', torch.zeros(num_lstm_layers, hidden_size))

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers, dropout=dropout)
        self.input_dropout = input_dropout

        if forget_bias is not None:
            initialize_forget_gate_bias(self.lstm, forget_bias)

    def initial_state(self, initial_value=None):
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
            output = output[0, 0, :]
            hx = hx.squeeze(1)
            cx = cx.squeeze(1)
        else:
            start = self.input_zeros
            hx = self.hidden_zeros
            cx = self.hidden_zeros
            # for the initial state, last layer hx is hx[-1, 0, :]
            output = hx[-1]

        return TreeStack(value=Node(initial_value, hx, cx, output), parent=None, length=1)

    def push_states(self, stacks, values, inputs):
        """
        Starting from a list of current stacks, put the inputs through the LSTM and build new stack nodes.

        B = stacks.len() = values.len()

        inputs must be of shape 1 x B x input_size
        """
        inputs = self.input_dropout(inputs)

        hx = torch.stack([t.value.lstm_hx for t in stacks], dim=1)
        cx = torch.stack([t.value.lstm_cx for t in stacks], dim=1)
        output, (hx, cx) = self.lstm(inputs, (hx, cx))

        # output shape: (1, batch, hidden_size) since input is one step
        # squeeze the time dimension once rather than slicing per node
        last_layer = output[0]  # (batch, hidden_size)

        new_stacks = [stack.push(Node(transition, hx[:, i, :], cx[:, i, :], last_layer[i]))
                      for i, (stack, transition) in enumerate(zip(stacks, values))]
        return new_stacks

    def output(self, stack):
        """
        Return the last layer of the lstm_hx as the output from a stack

        Refactored so that alternate structures have an easy way of getting the output
        """
        return stack.value.output
