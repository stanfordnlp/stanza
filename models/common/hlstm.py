import torch
import torch.nn as nn
import torch.nn.functional as F

class HLSTMCell(nn.modules.rnn.RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(HLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstmcell = nn.LSTMCell(input_size, hidden_size, bias=bias)

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

        hx = self.lstmcell(input, hx)

        gate = F.sigmoid(self.gate(torch.cat([c_l_minus_one, hx[1], input], 1)))

        return hx[0], hx[1] + gate * c_l_minus_one

if __name__ == "__main__":
    rnn = HLSTMCell(10, 20)
    rnn2 = HLSTMCell(20, 20)
    input = torch.randn(6, 3, 10)
    hx = torch.randn(3, 20)
    cx = torch.randn(3, 20)
    hx2 = torch.randn(3, 20)
    cx2 = torch.randn(3, 20)
    output = []
    for i in range(6):
        hx, cx = rnn(input[i], None, (hx, cx))
        hx2, cx2 = rnn2(hx, cx, (hx2, cx2))
        output.append(hx2)
    print(torch.stack(output))
