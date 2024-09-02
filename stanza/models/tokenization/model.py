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
        self.conv = nn.Conv1d(hidden_dim, hidden_dim,
                              args["sentence_analyzer_kernel"], padding="same",
                              padding_mode="circular")
        self.ffnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        # map the vocab to pretrain IDs
        embs = self.embeddings(torch.tensor([[self.vocab[j] for j in i] for i in x],
                                           device=self.device))
        net = self.emb_proj(embs)
        net = self.conv(net.permute(0,2,1)).permute(0,2,1)
        return self.ffnn(net)


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

        self.dropout = nn.Dropout(dropout)
        self.dropout_feat = nn.Dropout(feat_dropout)

        self.toknoise = nn.Dropout(self.args['tok_noise'])

    def forward(self, x, feats, text):
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

        # use the rough predictions from the char tokenizer to create word tokens
        # then use those word tokens + contextual/fixed word embeddings to refine
        # sentence predictions
        if self.args["sentence_second_pass"]:
            # these are the draft predictions for only token-level decisinos
            # which we can use to slice the text
            draft_preds = torch.cat([nontok, tok+nonmwt, tok+mwt], 2).argmax(dim=2)
            draft_preds = (draft_preds > 0)
            # we add a prefix zero
            # TODO inefficient / how to parallelize this?
            token_locations = [[-1] + i.nonzero().squeeze(1).cpu().tolist()
                               for i in draft_preds]

            # both: batch x seq x [variable: text token count]
            batch_tokens = [] # str tokens 
            batch_tokenid_locations = [] # id locations for the *end* of each str token
                                         # corresponding to char token
            for location,chars, toks in zip(token_locations, text, x):
                # we append len(chars)-1 to append  the last token which wouldn't
                # necessearily have been captured by the splits; though in theory
                # the model should put a token at the end of each sentence so this
                # should be less of a problem

                a,b = tee(location+[len(chars)-1])
                tokens = []
                tokenid_locations = []
                next(b) # because we want to start iterating on the NEXT id to create pairs
                j = -1
                for i,j in zip(a,b):
                    split = chars[i+1:j+1]
                    # if the entire unit is UNK, leave as UNK into the predictor
                    is_unk = ((toks[i+1:j+1]) == UNK_ID).all().cpu().item()
                    if set(split) == set([PAD]):
                        continue
                    tokenid_locations.append(j)

                    if not is_unk:
                        tokens.append("".join(split).replace(PAD, ""))
                    else:
                        tokens.append(UNK)

                batch_tokens.append(tokens)
                batch_tokenid_locations.append(tokenid_locations)

            # dynamically pad the batch tokens to size
            # why max 5? our 
            max_size = max(max([len(i) for i in batch_tokens]),
                           self.args["sentence_analyzer_kernel"])
            batch_tokens_padded = []
            batch_tokens_isntpad = []
            for i in batch_tokens:
                batch_tokens_padded.append(i + [PAD for _ in range(max_size-len(i))])
                batch_tokens_isntpad.append([True for _ in range(len(i))] +
                                            [False for _ in range(max_size-len(i))])

            ##### TODO EVERYTHING BELOW THIS LINE IS UNTESTED #####
            second_pass_scores = self.sent_2nd_pass_clf(batch_tokens_padded)

            # we only add scores for slots for which we have a possible word ending
            # i.e. its not padding and its also not a middle of rough score's resulting
            # words
            second_pass_chars_align = torch.zeros_like(sent0)
            token_location_selectors = torch.tensor([[i,k] for i,j in
                                                     enumerate(batch_tokenid_locations)
                                                     for k in j])

            second_pass_chars_align[
                token_location_selectors[:,0],
                token_location_selectors[:,1]
            ] = second_pass_scores[torch.tensor(batch_tokens_isntpad)]

            sent0 += second_pass_chars_align

        nonsent = F.logsigmoid(-sent0)
        sent = F.logsigmoid(sent0)

        if self.args['use_mwt']:
            pred = torch.cat([nontok, tok+nonsent+nonmwt, tok+sent+nonmwt, tok+nonsent+mwt, tok+sent+mwt], 2)
        else:
            pred = torch.cat([nontok, tok+nonsent, tok+sent], 2)

        return pred
