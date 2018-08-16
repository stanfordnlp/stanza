import torch
import torch.nn as nn
import torch.nn.functional as F

# Highway LSTM Cell (Zhang et al. (2018) Highway Long Short-Term Memory RNNs for Distant Speech Recognition)
class HLSTMCell(nn.modules.rnn.RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(HLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM parameters
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.Wg = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)

        # highway gate parameters
        self.gate = nn.Linear(input_size + 2 * hidden_size, hidden_size, bias=bias)

    def forward(self, input, c_l_minus_one=None, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        if c_l_minus_one is None:
            c_l_minus_one = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        self.check_forward_hidden(input, c_l_minus_one, 'c_l_minus_one')

        # vanilla LSTM computation
        rec_input = torch.cat([input, hx[0]], 1)
        i = F.sigmoid(self.Wi(rec_input))
        f = F.sigmoid(self.Wi(rec_input))
        o = F.sigmoid(self.Wi(rec_input))
        g = F.tanh(self.Wi(rec_input))

        # highway gates
        gate = F.sigmoid(self.gate(torch.cat([c_l_minus_one, hx[1], input], 1)))

        c = gate * c_l_minus_one + f * hx[1] + i * g
        h = o * F.tanh(c)

        return h, c

# Highway LSTM network, does NOT use the HLSTMCell above
class HighwayLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, recur_dropout=0, highway_func=None):
        super(HighwayLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.highway_func = highway_func
        self.recur_dropout = recur_dropout

        self.lstm = nn.ModuleList()
        self.highway = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.recur_drop = nn.Dropout(recur_dropout)
        if self.recur_dropout == 0:
            in_size = input_size
            for l in range(num_layers):
                self.lstm.append(nn.LSTM(in_size, hidden_size, num_layers=1, bias=bias,
                    batch_first=batch_first, dropout=0, bidirectional=bidirectional))
                self.highway.append(nn.Linear(in_size, hidden_size * self.num_directions, bias=bias))
                in_size = hidden_size * self.num_directions
        else:
            raise NotImplementedError()

    def forward(self, input, hx=None):
        highway_func = lambda x: x if self.highway_func is None else self.highway_func

        if self.recur_dropout == 0:
            hs = []
            cs = []
            for l in range(self.num_layers):
                layer_hx = (hx[0][l * self.num_directions:(l+1)*self.num_directions], hx[1][l * self.num_directions:(l+1)*self.num_directions]) if hx is not None else None
                h, _ = self.lstm[l](input, layer_hx)
                h = self.drop(h) + highway_func(self.highway[l](input))
                input = h
            return input
        else:
            raise NotImplementedError()

if __name__ == "__main__":
    T = 10
    bidir = True
    num_dir = 2 if bidir else 1
    rnn = HighwayLSTM(10, 20, num_layers=2, bidirectional=True)
    input = torch.randn(T, 3, 10)
    hx = torch.randn(2 * num_dir, 3, 20)
    cx = torch.randn(2 * num_dir, 3, 20)
    output = rnn(input, (hx, cx))
    print(output)
