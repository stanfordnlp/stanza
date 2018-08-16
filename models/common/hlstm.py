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

if __name__ == "__main__":
    T = 10
    rnn = HLSTMCell(10, 20)
    rnn2 = HLSTMCell(20, 20)
    input = torch.randn(T, 3, 10)
    hx = torch.randn(3, 20)
    cx = torch.randn(3, 20)
    hx2 = torch.randn(3, 20)
    cx2 = torch.randn(3, 20)
    output = []
    for i in range(T):
        hx, cx = rnn(input[i], None, (hx, cx))
        hx2, cx2 = rnn2(hx, cx, (hx2, cx2))
        output.append(hx2)
    print(torch.stack(output))
