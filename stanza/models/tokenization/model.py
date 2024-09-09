import torch
import torch.nn.functional as F
import torch.nn as nn
from itertools import tee

from stanza.models.common.seq2seq_constant import PAD, UNK, UNK_ID

class SentenceAnalyzer(nn.Module):
    def __init__(self, args, pretrain, hidden_dim, device=None):
        super().__init__()

        assert pretrain != None, "2nd pass sentence anayzer is missing pretrain word vectors"

        self.args = args
        self.vocab = pretrain.vocab
        self.embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(pretrain.emb), freeze=True)

        self.emb_proj = nn.Linear(pretrain.emb.shape[1], hidden_dim)
        self.tok_proj = nn.Linear(hidden_dim*2, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True,
                              batch_first=True, num_layers=args['rnn_layers'])

        # this is zero-initialized to make the second pass initially the id
        # function; and then it could change only as needed but would otherwise
        # be zero
        self.final_proj = nn.Parameter(torch.zeros(hidden_dim*2, 1), requires_grad=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        # map the vocab to pretrain IDs
        token_ids = [[self.vocab[j.strip()] for j in i] for i in x]
        embs = self.embeddings(torch.tensor(token_ids, device=self.device))
        net = self.emb_proj(embs) 
        net = self.lstm(net)[0]
        return self.final_proj @ net


class Tokenizer(nn.Module):
    def __init__(self, args, nchars, emb_dim, hidden_dim, dropout, feat_dropout, pretrain=None):
        super().__init__()

        self.args = args
        self.pretrain = pretrain
        feat_dim = args['feat_dim']

        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)

        self.rnn = nn.LSTM(emb_dim + feat_dim, hidden_dim, num_layers=self.args['rnn_layers'], bidirectional=True, batch_first=True, dropout=dropout if self.args['rnn_layers'] > 1 else 0)

        if self.args['conv_res'] is not None:
            self.conv_res = nn.ModuleList()
            self.conv_sizes = [int(x) for x in self.args['conv_res'].split(',')]

            for si, size in enumerate(self.conv_sizes):
                l = nn.Conv1d(emb_dim + feat_dim, hidden_dim * 2, size, padding=size//2, bias=self.args.get('hier_conv_res', False) or (si == 0))
                self.conv_res.append(l)

            if self.args.get('hier_conv_res', False):
                self.conv_res2 = nn.Conv1d(hidden_dim * 2 * len(self.conv_sizes), hidden_dim * 2, 1)
        self.tok_clf = nn.Linear(hidden_dim * 2, 1)
        self.sent_clf = nn.Linear(hidden_dim * 2, 1)
        if self.args['use_mwt']:
            self.mwt_clf = nn.Linear(hidden_dim * 2, 1)

        if args['hierarchical']:
            in_dim = hidden_dim * 2
            self.rnn2 = nn.LSTM(in_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.tok_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            self.sent_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            if self.args['use_mwt']:
                self.mwt_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)

        if args['sentence_second_pass']:
            self.sent_2nd_pass_clf = SentenceAnalyzer(args, pretrain, hidden_dim)
            # initially, don't use 2nd pass that much (this is near 0, meaning it will pretty much
            # not be mixed in
            self.sent_2nd_mix = nn.Parameter(torch.full((1,), -5.0), requires_grad=True)

        self.dropout = nn.Dropout(dropout)
        self.dropout_feat = nn.Dropout(feat_dropout)

        self.toknoise = nn.Dropout(self.args['tok_noise'])

    def forward(self, x, feats, text, detach_2nd_pass=False):
        emb = self.embeddings(x)
        emb = self.dropout(emb)
        feats = self.dropout_feat(feats)


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
        if self.args['use_mwt']:
            nonmwt = F.logsigmoid(-mwt0)
            mwt = F.logsigmoid(mwt0)

        nonsent = F.logsigmoid(-sent0)
        sent = F.logsigmoid(sent0)

        # use the rough predictions from the char tokenizer to create word tokens
        # then use those word tokens + contextual/fixed word embeddings to refine
        # sentence predictions

        if self.args["sentence_second_pass"]:
            # these are the draft predictions for only token-level decisinos
            # which we can use to slice the text
            if self.args['use_mwt']:
                draft_preds = torch.cat([nontok, tok+nonsent+nonmwt, tok+sent+nonmwt, tok+nonsent+mwt, tok+sent+mwt], 2).argmax(dim=2)
            else:
                draft_preds = torch.cat([nontok, tok+nonsent, tok+sent], 2).argmax(dim=2)

            draft_preds = (draft_preds > 0)
            # these boolean indicies are *inclusive*, so predict it or not
            # we need to split on the last token if we want to keep the
            # final word
            draft_preds[:,-1] = True

            # both: batch x [variable: text token count] 
            extracted_tokens = []
            partial = []
            last = 0
            last_batch = -1

            nonzero = draft_preds.nonzero().cpu().tolist()
            for i,j in nonzero:
                if i != last_batch:
                    last_batch = i
                    last = 0
                    if i != 0:
                        extracted_tokens.append(partial)
                    partial = []

                substring = text[i][last:j+1]
                last = j+1

                partial.append("".join(substring))
            extracted_tokens.append(partial)

            # dynamically pad the batch tokens to size
            # why to at least a fix size? it must be wider
            # than our kernel
            max_size = max(max([len(i) for i in extracted_tokens]),
                           self.args["sentence_analyzer_kernel"])
            batch_tokens_padded = []
            batch_tokens_isntpad = []
            for i in extracted_tokens:
                batch_tokens_padded.append(i + [PAD for _ in range(max_size-len(i))])
                batch_tokens_isntpad.append([True for _ in range(len(i))] +
                                            [False for _ in range(max_size-len(i))])

            pad_mask = torch.tensor(batch_tokens_isntpad)
            second_pass_scores = self.sent_2nd_pass_clf(batch_tokens_padded)

            # # we only add scores for slots for which we have a possible word ending
            # # i.e. its not padding and its also not a middle of rough score's resulting
            # # words
            second_pass_chars_align = torch.zeros_like(sent0)
            second_pass_chars_align[draft_preds] = second_pass_scores[pad_mask]

            mix = F.sigmoid(self.sent_2nd_mix)

            # update sent0 value
            if detach_2nd_pass:
                sent0 = (1-mix.detach())*sent0 + mix.detach()*second_pass_chars_align.detach()
            else:
                sent0 = (1-mix)*sent0 + mix*second_pass_chars_align

        nonsent = F.logsigmoid(-sent0)
        sent = F.logsigmoid(sent0)

        if self.args['use_mwt']:
            pred = torch.cat([nontok, tok+nonsent+nonmwt, tok+sent+nonmwt, tok+nonsent+mwt, tok+sent+mwt], 2)
        else:
            pred = torch.cat([nontok, tok+nonsent, tok+sent], 2)

        return pred
