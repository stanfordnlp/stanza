import torch
import torch.nn.functional as F
import torch.nn as nn

class Tokenizer(nn.Module):
    def __init__(self, args, nchars, emb_dim, hidden_dim, dropout, feat_dropout):
        super().__init__()

        self.args = args
        feat_dim = args['feat_dim']

        self.args['rnn_layers'] = 4

        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)
        self.embeddings2 = nn.Embedding(16264, emb_dim, padding_idx=0)

        self.rnn_syb = nn.LSTM(emb_dim, hidden_dim//2, num_layers=self.args['rnn_layers'], bidirectional=True, batch_first=True, dropout=dropout if self.args['rnn_layers'] > 1 else 0)
        self.rnn_char = nn.LSTM(emb_dim + feat_dim, hidden_dim//2, num_layers=self.args['rnn_layers'], bidirectional=True, batch_first=True, dropout=dropout if self.args['rnn_layers'] > 1 else 0)

        # This is for character embeddings:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        self.args['conv_res'] = "3,3,5,9"

        if self.args['conv_res'] is not None:
            self.conv_res = nn.ModuleList()
            self.conv_sizes = [int(x) for x in self.args['conv_res'].split(',')]

            for si, size in enumerate(self.conv_sizes):
                l = nn.Conv1d(hidden_dim, hidden_dim, size, padding=size//2, bias=self.args.get('hier_conv_res', False) or (si == 0))
                self.conv_res.append(l)

            if self.args.get('hier_conv_res', False):
                self.conv_res2 = nn.Conv1d(hidden_dim * 2 * len(self.conv_sizes), hidden_dim * 2, 1)
        self.tok_clf = nn.Linear(hidden_dim * 2, 1)
        self.sent_clf = nn.Linear(hidden_dim * 2, 1)
        if self.args['use_mwt']:
            self.mwt_clf = nn.Linear(hidden_dim * 2, 1)

        # Add separate layer for syllable embeddings:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        if self.args['conv_res'] is not None:
            self.syllable_conv_res = nn.ModuleList()
            self.conv_sizes = [3,3,5,9]
            for si, size in enumerate(self.conv_sizes):

                l = nn.Conv1d(hidden_dim, hidden_dim, size, padding=size//2, bias=self.args.get('hier_conv_res', False) or (si == 0))
                self.syllable_conv_res.append(l)

        if args['hierarchical']:
            in_dim = hidden_dim * 2
            self.rnn2 = nn.LSTM(in_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.tok_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            self.sent_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            if self.args['use_mwt']:
                self.mwt_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.dropout_feat = nn.Dropout(feat_dropout)

        self.toknoise = nn.Dropout(self.args['tok_noise'])

    def forward(self, x, x2, feats):

        emb = self.embeddings(x)
        emb = self.dropout(emb)

        emb2 = self.embeddings2(x2)
        emb2 = self.dropout(emb2)

        emb = torch.cat([emb, feats], 2)

        inp, _ = self.rnn_char(emb)
        inp2, _ = self.rnn_syb(emb2)

        #inp = torch.cat([inp, inp2], 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

        if self.args['conv_res'] is not None:
            conv_input = inp.transpose(1, 2).contiguous()
            syllable_input = inp2.transpose(1, 2).contiguous()
            if not self.args.get('hier_conv_res', False):

                conv1 = self.dropout(self.conv_res[0](conv_input).transpose(1, 2).contiguous())
                conv2 = self.dropout(self.conv_res[1](conv_input).transpose(1, 2).contiguous())
                conv3 = self.dropout(self.conv_res[2](conv_input).transpose(1, 2).contiguous())

                inp = torch.stack((conv1,conv2,conv3), 3)
                inp, _ = torch.max(inp, 3)

                conv1 = self.dropout(self.syllable_conv_res[0](syllable_input).transpose(1, 2).contiguous())
                conv2 = self.dropout(self.syllable_conv_res[1](syllable_input).transpose(1, 2).contiguous())
                conv3 = self.dropout(self.syllable_conv_res[2](syllable_input).transpose(1, 2).contiguous())

                inp2 = torch.stack((conv1,conv2,conv3), 3)
                inp2, _ = torch.max(inp2, 3)

                inp = torch.cat([inp, inp2], 2)

                """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                for l in self.conv_res:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    inp = inp + l(conv_input).transpose(1, 2).contiguous()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                """
            else:
                hid = []
                for l in self.conv_res:
                    hid += [l(conv_input)]
                hid = torch.cat(hid, 1)
                hid = F.relu(hid)
                hid = self.dropout(hid)
                inp = inp + self.conv_res2(hid).transpose(1, 2).contiguous()

        inp = self.dropout(inp)

        tok0 = self.tok_clf(inp)
        sent0 = self.sent_clf(inp)
        if self.args['use_mwt']:
            mwt0 = self.mwt_clf(inp)

        if self.args['hierarchical']:
            if self.args['hier_invtemp'] > 0:
                inp2, _ = self.rnn2(inp * (1 - self.toknoise(torch.sigmoid(-tok0 * self.args['hier_invtemp']))))
            else:
                inp2, _ = self.rnn2(inp)

            inp2 = self.dropout(inp2)

            tok0 = tok0 + self.tok_clf2(inp2)
            sent0 = sent0 + self.sent_clf2(inp2)
            if self.args['use_mwt']:
                mwt0 = mwt0 + self.mwt_clf2(inp2)

        nontok = F.logsigmoid(-tok0)
        tok = F.logsigmoid(tok0)
        nonsent = F.logsigmoid(-sent0)
        sent = F.logsigmoid(sent0)
        if self.args['use_mwt']:
            nonmwt = F.logsigmoid(-mwt0)
            mwt = F.logsigmoid(mwt0)

        if self.args['use_mwt']:
            pred = torch.cat([nontok, tok+nonsent+nonmwt, tok+sent+nonmwt, tok+nonsent+mwt, tok+sent+mwt], 2)
        else:
            pred = torch.cat([nontok, tok+nonsent, tok+sent], 2)

        y[y==-1] = 0
        y[y==1] = 0
        y[y==2] = 1
        word_mask = x.gt(0)
        sent_pred = torch.cat([nontok, tok+nonsent, tok+sent], 2)

        loss, trans = self.crit(sent_pred, word_mask, y)

        return pred, loss, trans












