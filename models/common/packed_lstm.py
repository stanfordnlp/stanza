import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

class PackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, pad=False, rec_dropout=0):
        super().__init__()

        self.batch_first = batch_first
        self.pad = pad
        if rec_dropout == 0:
            # use the fast, native LSTM implementation
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        else:
            self.lstm = LSTMwRecDropout(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, rec_dropout=rec_dropout)

    def forward(self, input, mask, hx=None):
        lengths = mask.size(1) - mask.sum(1)
        input = pack_padded_sequence(input, lengths, batch_first=self.batch_first)

        res = self.lstm(input, hx)
        if self.pad:
            res = (pad_packed_sequence(res[0], batch_first=self.batch_first)[0], res[1])
        return res

class LSTMwRecDropout(nn.Module):
    """ An LSTM implementation that supports recurrent dropout """
    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, pad=False, rec_dropout=0):
        super().__init__()
        self.batch_first = batch_first
        self.pad = pad
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.rec_dropout = nn.Dropout(rec_dropout)

        self.num_directions = 2 if bidirectional else 1

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else self.num_directions * hidden_size
            for d in range(self.num_directions):
                self.cells.append(nn.LSTMCell(in_size, hidden_size, bias=bias))

    def forward(self, input, hx=None):
        def rnn_loop(x, cell, init, reverse=False):
            # RNN loop for one layer in one direction with recurrent dropout
            # Assumes input is PackedSequence, returns PackedSequence as well
            batch_size = x.batch_sizes[0].item()
            states = [[i for _ in range(batch_size)] for i in init]
            h_drop_mask = x.data.new_ones(batch_size, self.hidden_size)
            h_drop_mask = self.rec_dropout(h_drop_mask)
            resh = [[] for _ in range(batch_size)]

            if not reverse:
                st = 0
                for bs in x.batch_sizes:
                    s1 = cell(x.data[st:st+bs], (torch.cat(states[0][:bs], 0) * h_drop_mask[:bs], torch.cat(states[1][:bs], 0)))
                    for j in range(bs):
                        resh[j].append(s1[0][j].unsqueeze(0))
                        states[0][j] = s1[0][j].unsqueeze(0)
                        states[1][j] = s1[1][j].unsqueeze(0)
                    st += bs
            else:
                en = x.data.size(0).item()
                for bs in reversed(x.batch_sizes):
                    s1 = cell(x.data[en-bs:en], (torch.cat(states[0][:bs], 0) * h_drop_mask[:bs], torch.cat(states[1][:bs], 0)))
                    for j in range(bs):
                        resh[j] = [s1[0][j].unsqueeze(0)] + resh[j]
                        states[0][j] = s1[0][j].unsqueeze(0)
                        states[1][j] = s1[1][j].unsqueeze(0)
                    en -= bs

            resh = pack_sequence([torch.cat(hs, 0) for hs in resh])
            return resh, tuple(torch.cat(s, 0) for s in states)

        all_states = [[], []]
        for l in range(self.num_layers):
            new_input = []
            for d in range(self.num_directions):
                idx = l * self.num_directions + d
                cell = self.cells[idx]
                out, states = rnn_loop(input, cell, (hx[i][idx].unsqueeze(0) for i in range(2)) if hx is not None else (input.data.new_zeros(1, self.hidden_size) for _ in range(2)))

                new_input.append(out)
                all_states[0].append(states[0].unsqueeze(0))
                all_states[1].append(states[1].unsqueeze(0))

            if self.num_directions > 1:
                # concatenate both directions
                input = PackedSequence(self.dropout(torch.cat([x.data for x in new_input], 1)), new_input[0].batch_sizes)
            else:
                input = PackedSequence(self.dropout(new_input[0].data), new_input[0].batch_sizes)

        return input, tuple(torch.cat(x, 0) for x in all_states)
