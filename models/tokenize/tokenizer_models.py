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
    def __init__(self, args, nchars, emb_dim, hidden_dim, N_CLASSES=5, dropout=0):
        super().__init__()

        self.args = args
        feat_dim = args['feat_dim']

        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)

        self.rnn = nn.LSTM(emb_dim + feat_dim, hidden_dim, num_layers=self.args['rnn_layers'], bidirectional=True, batch_first=True, dropout=dropout if self.args['rnn_layers'] > 1 else 0)
        #self.rnn2 = nn.LSTM(emb_dim + feat_dim + 1, hidden_dim, num_layers=1, bidirectional=True)

        if self.args['conv_res'] is not None:
            self.conv_res = nn.ModuleList()
            self.conv_sizes = [int(x) for x in self.args['conv_res'].split(',')]

            for si, size in enumerate(self.conv_sizes):
                l = nn.Conv1d(emb_dim + feat_dim, hidden_dim * 2, size, padding=size//2, bias=self.args.get('hier_conv_res', False) or (si == 0))
                self.conv_res.append(l)

            if self.args.get('hier_conv_res', False):
                self.conv_res2 = nn.Conv1d(hidden_dim * 2 * len(self.conv_sizes), hidden_dim * 2, 1)
        #self.hid = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.tok_clf = nn.Linear(hidden_dim * 2, 1)
        self.sent_clf = nn.Linear(hidden_dim * 2, 1)
        self.mwt_clf = nn.Linear(hidden_dim * 2, 1)

        if args['hierarchical']:
            in_dim = hidden_dim * 2
            self.rnn2 = nn.LSTM(in_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
            #self.sent_clf = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 1, bias=False))
            #self.mwt_clf = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 1, bias=False))
            #self.hierhid = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.tok_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            self.sent_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            self.mwt_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.toknoise = nn.Dropout(self.args['tok_noise'])

    def forward(self, x, feats):
        emb = self.embeddings(x)

        emb = self.dropout(emb)

        emb = torch.cat([emb, feats], 2)

        inp, _ = self.rnn(emb)

        if self.args['conv_res'] is not None:
            conv_input = emb.transpose(1, 2).contiguous()
            if not self.args.get('hier_conv_res', False):
                for l in self.conv_res:
                    inp = inp + l(conv_input).transpose(1, 2).contiguous()
            else:
                hid = []
                for l in self.conv_res:
                    hid += [l(conv_input)]
                hid = torch.cat(hid, 1)
                hid = F.relu(hid)
                hid = self.dropout(hid)
                inp = inp + self.conv_res2(hid).transpose(1, 2).contiguous()

        inp = self.dropout(inp)

        #inp = self.hid(inp)
        #inp = self.dropout(F.relu(inp))
        tok0 = self.tok_clf(inp)
        sent0 = self.sent_clf(inp)
        mwt0 = self.mwt_clf(inp)

        if self.args['hierarchical']:
            inp2, _ = self.rnn2(inp * (1 - self.toknoise(F.sigmoid(-tok0 * self.args['hier_invtemp']))))

            inp2 = self.dropout(inp2)

            #inp2 = self.dropout(F.relu(self.hierhid(inp2)))
            tok0 = tok0 + self.tok_clf2(inp2)
            sent0 = sent0 + self.sent_clf2(inp2)
            mwt0 = mwt0 + self.mwt_clf2(inp2)

        nontok = F.logsigmoid(-tok0)
        tok = F.logsigmoid(tok0)
        nonsent = F.logsigmoid(-sent0)
        sent = F.logsigmoid(sent0)
        nonmwt = F.logsigmoid(-mwt0)
        mwt = F.logsigmoid(mwt0)

        pred = torch.cat([nontok, tok+nonsent+nonmwt, tok+sent+nonmwt, tok+nonsent+mwt, tok+sent+mwt], 2)

        return pred, []
