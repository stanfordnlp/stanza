"""
Dynamic Skip Connection LSTM

@article{gui2018long,
  title={Long Short-Term Memory with Dynamic Skip Connections},
  author={Gui, Tao and Zhang, Qi and Zhao, Lujun and Lin, Yaosong and Peng, Minlong and Gong, Jingjing and Huang, Xuanjing},
  journal={arXiv preprint arXiv:1811.03873},
  year={2018}
}

Adapted from https://github.com/v1xerunt/PyTorch-Dynamic_LSTM

MIT License

Copyright (c) 2019 v1xerunt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
from torch.autograd import Variable

class DynamicLSTM(nn.Module):
    # lambda is a keyword
    def __init__(self, n_actions, n_units, n_input, n_hidden, n_output, lamda=1, dropout=0.3):
        """
        n_actions: number of dynamic connections to make
        n_units:   number of channels to use when picking actions
        n_input:   number of channels in input
        n_hidden:  number of channels to use internally
        n_output:  number of channels to output
        """
        super(DynamicLSTM, self).__init__()

        # hyperparameters
        self.n_actions = n_actions  # last K hidden state
        self.n_units = n_units  # hidden unit of Agent MLP
        self.n_input = n_input  # input size
        self.n_hidden = n_hidden  # hidden size of LSTM
        self.n_output = n_output  # output dim
        self.lamda = lamda
        self.dropout = dropout

        # layers
        self.fc1 = nn.Linear(self.n_hidden + self.n_input, self.n_units)
        self.fc2 = nn.Linear(self.n_units, self.n_actions)
        self.x2h = nn.Linear(self.n_input, 4 * self.n_hidden)
        self.h2h = nn.Linear(self.n_hidden, 4 * self.n_hidden)
        self.output = nn.Linear(self.n_hidden, self.n_output)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()

    def choose_action(self, observation, cur_time, agent_action, agent_prob):
        observation = observation.detach()
        result_fc1 = self.fc1(observation)
        result_fc2 = self.fc2(result_fc1)
        probs = self.softmax(result_fc2)
        m = torch.distributions.Categorical(probs)
        actions = m.sample()
        if cur_time != 0:
            agent_action.append(actions.unsqueeze(-1))
            agent_prob.append(m.log_prob(actions))

        return actions.unsqueeze(-1)

    def forward(self, input, agent_action=None, agent_prob=None):
        # input shape [batch_size, timestep, feature_dim]
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        assert(feature_dim == self.n_input)

        # Initialization
        cur_time = 0  # Current timestep
        if agent_action is None:
            agent_action = []  # Actions for agents
        if agent_prob is None:
            agent_prob = []  # Probabilities for agents
        # Hidden state for lstm
        cur_h = Variable(torch.zeros(batch_size, self.n_hidden))
        # Cell memory for lstm
        cur_c = Variable(torch.zeros(batch_size, self.n_hidden))
        full_c = []  # Cell memory list for lstm
        full_h = []  # Hidden state list for lstm

        for cur_time in range(time_step):
            if cur_time == 0:
                self.choose_action(
                    torch.cat((input[:, 0, :], cur_h), 1), cur_time, agent_action, agent_prob)
                observed_c = torch.zeros_like(cur_c, dtype=torch.float32).view(-1).repeat(
                    self.n_actions).view(self.n_actions, batch_size, self.n_hidden)
                observed_h = torch.zeros_like(cur_h, dtype=torch.float32).view(-1).repeat(
                    self.n_actions).view(self.n_actions, batch_size, self.n_hidden)
                action_c = cur_c
                action_h = cur_h
            else:
                observed_c = torch.cat((observed_c[1:], cur_c.unsqueeze(0)), 0)
                observed_h = torch.cat((observed_h[1:], cur_h.unsqueeze(0)), 0)
                # use h(t-1) or mean h?
                observation = torch.cat((input[:, cur_time, :], cur_h), 1)
                actions = self.choose_action(observation, cur_time, agent_action, agent_prob)
                coord = torch.cat((actions.int(), torch.arange(
                    batch_size, dtype=torch.int).unsqueeze(-1)), 1)
                action_c = torch.stack([observed_c[i, j, :]
                                        for [i, j] in coord])
                action_h = torch.stack([observed_h[i, j, :]
                                        for [i, j] in coord])

            weighted_c = self.lamda * action_c + (1-self.lamda)*cur_c
            weighted_h = self.lamda * action_h + (1-self.lamda)*cur_h

            gates = self.x2h(input[:, cur_time, :]) + self.h2h(weighted_h)
            gates = gates.squeeze()

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = self.sigmoid(ingate)
            forgetgate = self.sigmoid(forgetgate)
            if self.dropout == 0:
                cellgate = self.tanh(cellgate)
            else:
                cellgate = self.dropout(self.tanh(cellgate))
            outgate = self.sigmoid(outgate)

            cur_c = torch.mul(weighted_c, forgetgate) + \
                torch.mul(ingate, cellgate)
            cur_h = torch.mul(outgate, self.tanh(cur_c))
            full_c.append(cur_c)
            full_h.append(cur_h)

        opt = self.output(cur_h)
        opt = self.softmax(opt)

        return opt, (agent_action, agent_prob)
