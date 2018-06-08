import torch
import torch.nn.functional as F
import torch.nn as nn

class Tokenizer(nn.Module):
    def __init__(self, args, nchars, emb_dim, hidden_dim, N_CLASSES=4, dropout=0):
        super().__init__()

        self.args = args
        feat_dim = args['feat_dim']

        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)

        self.conv_sizes = [[int(y) for y in x.split(',')] for x in args['conv_filters'].split(',,')]

        self.conv_filters = nn.ModuleList()

        if args['residual']:
            self.res_filters = nn.ModuleList()

        if args['aux_clf'] > 0 or args['merge_aux_clf']:
            self.aux_clf = nn.ModuleList()

        for layer, sizes in enumerate(self.conv_sizes):
            thislayer = nn.ModuleList()
            in_dim = emb_dim + feat_dim if layer == 0 else len(self.conv_sizes[layer-1]) * hidden_dim
            for size in sizes:
                thislayer.append(nn.Conv1d(in_dim, hidden_dim, size, padding=size//2))

            self.conv_filters.append(thislayer)

            if args['residual']:
                self.res_filters.append(nn.Conv1d(in_dim, len(self.conv_sizes[layer]) * hidden_dim, 1))

            if (args['aux_clf'] > 0 or args['merge_aux_clf']) and layer < len(self.conv_sizes) - 1:
                self.aux_clf.append(nn.Conv1d(hidden_dim * len(self.conv_sizes[layer]), N_CLASSES, 1))

        self.dense_clf = nn.Conv1d(hidden_dim * len(self.conv_sizes[-1]), N_CLASSES, 1)

        self.dropout = nn.Dropout(dropout)
        self.input_dropout = nn.Dropout(dropout)

    def forward(self, x, feats):
        emb = self.embeddings(x)

        if self.args['input_dropout']:
            emb = self.input_dropout(emb)

        emb = torch.cat([emb, feats], 2)

        emb = emb.transpose(1, 2).contiguous()
        inp = emb

        aux_outputs = []

        for layeri, layer in enumerate(self.conv_filters):
            out = [f(inp) for f in layer]
            out = torch.cat(out, 1)
            out = F.relu(out)
            if self.args['residual']:
                out += self.res_filters[layeri](inp)
            if (self.args['aux_clf'] > 0 or self.args['merge_aux_clf']) and layeri < len(self.conv_sizes) - 1:
                aux_outputs += [self.aux_clf[layeri](out).transpose(1, 2).contiguous()]
            if layeri < len(self.conv_filters) - 1:
                out = self.dropout(out)
            inp = out

        pred = self.dense_clf(inp)
        pred = pred.transpose(1, 2).contiguous()

        if self.args['merge_aux_clf']:
            for aux_out in aux_outputs:
                pred += aux_out
            pred /= len(self.conv_sizes)

        return pred, aux_outputs

class RNNTokenizer(nn.Module):
    def __init__(self, args, nchars, emb_dim, hidden_dim, N_CLASSES=4, dropout=0):
        super().__init__()

        self.args = args
        feat_dim = args['feat_dim']

        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)

        self.rnn = nn.LSTM(emb_dim + feat_dim, hidden_dim, num_layers=self.args['rnn_layers'], bidirectional=True, batch_first=True, dropout=dropout if self.args['rnn_layers'] > 1 else 0)
        #self.rnn2 = nn.LSTM(emb_dim + feat_dim + 1, hidden_dim, num_layers=1, bidirectional=True)

        if self.args['conv_res'] is not None:
            self.conv_res = nn.ModuleList()
            self.conv_sizes = [int(x) for x in self.args['conv_res'].split(',')]

            for size in self.conv_sizes:
                self.conv_res.append(nn.Conv1d(emb_dim + feat_dim, hidden_dim * 2, size, padding=size//2))

        self.dense_clf = nn.Linear(hidden_dim * 2, N_CLASSES)

        if args['hierarchical']:
            self.rnn2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.dense_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, feats):
        emb = self.embeddings(x)

        emb = self.dropout(emb)

        emb = torch.cat([emb, feats], 2)

        inp, _ = self.rnn(emb)

        if self.args['conv_res'] is not None:
            conv_input = emb.transpose(1, 2).contiguous()
            for l in self.conv_res:
                inp = inp + l(conv_input).transpose(1, 2).contiguous()

        inp = self.dropout(inp)

        pred0 = self.dense_clf(inp)

        if self.args['hierarchical']:
            pred0_ = F.log_softmax(pred0, 2)

            #emb = torch.cat([emb, pred0_[:,:,0].unsqueeze(2)], 2)
            inp2, _ = self.rnn2(inp * (1 - torch.exp(pred0_[:,:,0].unsqueeze(2))))
            inp2 = self.dropout(inp2)

            pred1 = self.dense_clf2(inp2)

            pred = torch.cat([pred0[:,:,:2], pred0[:,:,2].unsqueeze(2) + pred1, pred0[:,:,3].unsqueeze(2)], 2)
        else:
            pred = pred0

        return pred, []
