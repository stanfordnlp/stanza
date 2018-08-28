import torch
import torch.nn as nn
import torch.nn.functional as F

class PackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, pad=False):
        super().__init__()

        self.batch_first = batch_first
        self.pad = pad
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input, mask, hx=None):
        lengths = mask.size(1) - mask.sum(1)
        input = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=self.batch_first)

        res = self.lstm(input, hx)
        if self.pad:
            res = (nn.utils.rnn.pad_packed_sequence(res[0], batch_first=self.batch_first)[0], res[1])
        return res
