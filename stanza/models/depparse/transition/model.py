from enum import Enum

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from stanza.models.common.utils import build_nonlinearity, unsort
from stanza.models.common.vocab import VOCAB_PREFIX_SIZE
from stanza.models.depparse.model import BaseParser
from stanza.models.depparse.transition.state import state_from_text, states_from_heads, TransitionLSTMEmbedding, SubtreeLSTMEmbedding
from stanza.models.depparse.transition.transitions import Shift, Finalize, ProjectiveLeft, ProjectiveRight, NonprojectiveLeft, NonprojectiveRight

# A few notes on some experiments crossvalidating the hyperparameters for this model
#
# On some experiments with the original combination method
#   (no transformer of the word vectors):
# Adam topped out around 87 LAS dev set
# AdaDelta seems to work better
# LAS dev score:
#    LR     WD       LAS
#   1.0   0.02     19.16
#   1.0   0.01     32.27
#   1.0   0.002    73.98
#   1.0   0.001    80.18
#   1.0   0.0005   83.54
#   1.0   0.0002   87.04
#   1.0   0.0001   87.42
#   1.0   0.00005  87.65
#   1.0   0.00002  87.76
#   1.0   0.00001  87.81
#   1.0   0.000005 87.86
#
#   2.0   0.002    74.84
#   2.0   0.0005   84.59
#   2.0   0.0002   86.99
#   2.0   0.00005  87.84
#   2.0   0.00002  87.92
#   2.0   0.00001  88.07
#   2.0   0.000005 88.05
#
#   5.0   0.0002   85.41
#   5.0   0.0001   87.39
#   5.0   0.00005  87.35
#   5.0   0.00002  88.01
#   5.0   0.00001  87.36
#   5.0   0.000005 87.54
#
# Continued training from one of the base models, using Adam with the default beta:
#
#    lr     dev
# 0.00003  0.8884
# 0.00005  0.8881
# 0.00008  0.8895
# 0.0001   0.8893
# 0.0002   0.8895
# 0.0003   0.8892
# 0.0005   0.8865
# 0.001    0.8865
# 0.002    0.8817
# 0.003    0.8760
# 0.005    0.8609
# 0.01     0.7893
# 0.02     0.6428
#
# Using beta=0.999 (like the pytorch default) might be slightly better:
#
#   lr      dev
# 0.00008  88.93
# 0.0001   89.01
# 0.0002   89.05
# 0.0003   88.90
#
# We experimented with a wide variety of subtree combination methods,
# but unfortunately the one that worked the best was simply passing through
# the original embedding of the word for the embedding of the new subtree

class SubtreeCombination(Enum):
    NONE             = 1
    LINEAR           = 2
    HEAD_LINEAR      = 3
    UNTIED_LINEAR    = 4
    BILINEAR         = 5
    MAX              = 6
    LSTM             = 7
    BILSTM           = 8

class TransitionParser(BaseParser):
    def __init__(self, args, vocab, emb_matrix=None, foundation_cache=None, bert_model=None, bert_tokenizer=None, force_bert_saved=False, peft_name=None):
        super().__init__(args, vocab, emb_matrix=emb_matrix, foundation_cache=foundation_cache, bert_model=bert_model, bert_tokenizer=bert_tokenizer, force_bert_saved=force_bert_saved, peft_name=peft_name)

        # recurrent layers
        # the parserlstm is already defined in the BaseParser
        relations = list(self.vocab['deprel'])
        self.relations = [x for x in relations[VOCAB_PREFIX_SIZE:] if x != 'root']
        self.relation_to_id = {x: idx for idx, x in enumerate(self.relations)}

        self.transitions = [Shift(), Finalize()] + [ProjectiveLeft(deprel) for deprel in self.relations] + [ProjectiveRight(deprel) for deprel in self.relations]
        self.transition_to_id = {x: idx for idx, x in enumerate(self.transitions)}
        self.transition_embedding_dim = self.args['transition_embedding_dim']
        self.transition_hidden_dim = self.args['transition_hidden_dim']
        self.transition_embedding = nn.Embedding(num_embeddings = len(self.transitions),
                                                 embedding_dim = self.transition_embedding_dim)
        self.transition_lstm = nn.LSTM(input_size=self.transition_embedding_dim, hidden_size=self.transition_hidden_dim, num_layers=self.args['num_layers'], dropout=self.args['dropout'])
        # random note: this does properly train when built from a .zeros()
        self.transition_start = nn.Parameter(torch.zeros(self.args['transition_hidden_dim']))
        self.transition_h0 = nn.Parameter(torch.zeros(self.args['num_layers'], self.args['transition_hidden_dim']))
        self.transition_c0 = nn.Parameter(torch.zeros(self.args['num_layers'], self.args['transition_hidden_dim']))

        self.partial_tree_lstm = nn.LSTM(input_size=self.args['hidden_dim'] * 2, hidden_size=self.args['hidden_dim'], num_layers=self.args['num_layers'], dropout=self.args['dropout'])
        self.partial_tree_start = nn.Parameter(torch.zeros(self.args['hidden_dim'] * 2))
        self.partial_tree_h0 = nn.Parameter(torch.zeros(self.args['num_layers'], self.args['hidden_dim']))
        self.partial_tree_c0 = nn.Parameter(torch.zeros(self.args['num_layers'], self.args['hidden_dim']))

        # the bidirectional LSTM is x2, adding in the partial trees is another x1
        self.final_hidden_dim = self.transition_hidden_dim + self.args['hidden_dim'] * 3
        self.output_layers = nn.ModuleList([nn.Linear(self.final_hidden_dim, self.final_hidden_dim)])

        self.nonlinearity = nn.ReLU()
        self.transition_subtree_nonlinearity = build_nonlinearity(self.args.get('transition_subtree_nonlinearity'))

        self.output_basic = nn.Linear(self.final_hidden_dim, 2)
        self.output_left = nn.Linear(self.final_hidden_dim, 1)
        self.output_right = nn.Linear(self.final_hidden_dim, 1)
        # this will be used to predict the relation if a transition is chosen
        self.output_deprel = nn.Linear(self.final_hidden_dim, len(self.relations))

        # TODO: maybe make an attention layer?
        # maybe split this across different relations or right/left?
        self.merge_words = nn.Linear(self.args['hidden_dim'] * 4, self.args['hidden_dim'] * 2)

        # TODO: again, left/right or include a relation embedding
        if self.args['transition_subtree_combination'] in (SubtreeCombination.LINEAR, SubtreeCombination.HEAD_LINEAR):
            self.merge_subtrees = nn.Linear(self.args['hidden_dim'] * 4, self.args['hidden_dim'] * 2)
        elif self.args['transition_subtree_combination'] is SubtreeCombination.UNTIED_LINEAR:
            self.merge_subtrees = nn.ModuleDict()
            for relation in self.relations:
                self.merge_subtrees[relation] = nn.Linear(self.args['hidden_dim'] * 4, self.args['hidden_dim'] * 2)
        elif self.args['transition_subtree_combination'] is SubtreeCombination.BILINEAR:
            self.merge_subtrees = nn.Bilinear(self.args['hidden_dim'] * 2, self.args['hidden_dim'] * 2, self.args['hidden_dim'] * 2)
        elif self.args['transition_subtree_combination'] is SubtreeCombination.MAX:
            self.reduce_linear = nn.Linear(self.args['hidden_dim'] * 2, self.args['hidden_dim'] * 2)
        elif self.args['transition_subtree_combination'] is SubtreeCombination.NONE:
            pass
        elif self.args['transition_subtree_combination'] is SubtreeCombination.LSTM:
            self.reduce_lstm = nn.LSTM(input_size=self.args['hidden_dim'] * 2, hidden_size=self.args['hidden_dim'] * 2, num_layers=self.args['num_layers'], dropout=self.args['dropout'])
            self.reduce_relation_embedding = nn.Embedding(num_embeddings = len(self.relations),
                                                          embedding_dim = self.args['hidden_dim'] * 2)
        elif self.args['transition_subtree_combination'] is SubtreeCombination.BILSTM:
            self.reduce_lstm = nn.LSTM(input_size=self.args['hidden_dim'] * 2, hidden_size=self.args['hidden_dim'], num_layers=self.args['num_layers'], dropout=self.args['dropout'], bidirectional=True)
            self.reduce_relation_embedding = nn.Embedding(num_embeddings = len(self.relations),
                                                          embedding_dim = self.args['hidden_dim'] * 2)
        else:
            raise ValueError("Unknown transition_subtree_combination %s" % self.args['transition_subtree_combination'])

        self.transition_loss_function = nn.CrossEntropyLoss(reduction='sum')
        self.deprel_loss_function = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, states):
        """
        Builds a list of logits for the different operations, including a separate one for each Left and Right merge 

        The return is:
          list of logits for each state
          list of list of left deprel scores for each state, each possible merge
          list of list of right deprel scores for each state, each possible merge
        """
        device = next(self.parameters()).device

        # first, we build the transition embeddings LSTM input
        # the states each keep track of the embeddings and most recent
        # output from the transition LSTM
        # in this way, when concatenating a new transition, we just need
        # to run one LSTM cell, rather than rerunning the whole LSTM
        transition_embeddings = []
        transition_h0 = []
        transition_c0 = []
        for state in states:
            if len(state.transitions) == 0:
                transition_embeddings.append(self.transition_start)
                transition_h0.append(self.transition_h0)
                transition_c0.append(self.transition_c0)
            else:
                transition = state.transitions[-1]
                if isinstance(transition, NonprojectiveRight):
                    transition = ProjectiveRight(transition.deprel)
                elif isinstance(transition, NonprojectiveLeft):
                    transition = ProjectiveLeft(transition.deprel)
                transition_id = torch.tensor(self.transition_to_id[transition], requires_grad=False, dtype=torch.long, device=device)
                transition_emb = self.transition_embedding(transition_id)
                transition_embeddings.append(transition_emb)
                transition_h0.append(state.transition_lstm_embeddings[-1].h0)
                transition_c0.append(state.transition_lstm_embeddings[-1].c0)
        transition_embeddings = torch.stack(transition_embeddings, dim=0).unsqueeze(0)
        transition_h0 = torch.stack(transition_h0, dim=1)
        transition_c0 = torch.stack(transition_c0, dim=1)
        transition_embeddings, (transition_h0, transition_c0) = self.transition_lstm(transition_embeddings, (transition_h0, transition_c0))
        transition_embeddings = transition_embeddings.squeeze(0)

        #print("------------------")
        partial_tree_embeddings = []
        partial_tree_h0 = []
        partial_tree_c0 = []
        for state in states:
            if len(state.subtree_lstm_embeddings) == 0:
                partial_tree_embeddings.append(self.partial_tree_start)
                partial_tree_h0.append(self.partial_tree_h0)
                partial_tree_c0.append(self.partial_tree_c0)
            else:
                head = state.current_heads[-1]
                partial_tree_embeddings.append(state.subtree_embeddings[head])
                partial_tree_h0.append(state.subtree_lstm_embeddings[-1].h0)
                partial_tree_c0.append(state.subtree_lstm_embeddings[-1].c0)
            #print("Incremental partial tree:", torch.linalg.norm(partial_tree_embeddings[-1]))
        partial_tree_embeddings = torch.stack(partial_tree_embeddings, dim=0).unsqueeze(0)
        partial_tree_h0 = torch.stack(partial_tree_h0, dim=1)
        partial_tree_c0 = torch.stack(partial_tree_c0, dim=1)
        partial_tree_embeddings, (partial_tree_h0, partial_tree_c0) = self.partial_tree_lstm(partial_tree_embeddings, (partial_tree_h0, partial_tree_c0))
        partial_tree_embeddings = partial_tree_embeddings.squeeze(0)
        #print(torch.linalg.norm(partial_tree_embeddings))

        word_embeddings = [state.word_embeddings[state.word_position] for state in states]
        word_embeddings = torch.stack(word_embeddings)
        output_hx = torch.cat([transition_embeddings, partial_tree_embeddings, word_embeddings], dim=1)
        for output_layer in self.output_layers:
            # TODO: dropout?
            output_hx = self.nonlinearity(output_hx)
            output_hx = output_layer(output_hx)
        # batch size x 2 - Shift or Finalize
        basic_output = self.output_basic(self.nonlinearity(output_hx))
        final_output = [[x] for x in basic_output]
        left_deprels = []
        right_deprels = []

        for state_idx, state in enumerate(states):
            left_deprel = None
            right_deprel = None
            if len(state.current_heads) > 1:
                # TODO: add a position embedding for the projective / non-projective attachments?
                attachment_embeddings = [torch.cat([state.subtree_embeddings[x], state.subtree_embeddings[state.current_heads[-1]]])
                                         for x in range(1, state.word_position+1)]
                attachment_embeddings = [self.merge_words(x) for x in attachment_embeddings]
                attachment_embeddings = torch.stack(attachment_embeddings, dim=0)

                # in addition to the current words, we also use the current transition and partial tree
                # LSTM outputs to determine the scores of each attachment (and possible dependency)
                attachment_input = torch.cat([transition_embeddings[state_idx, :], partial_tree_embeddings[state_idx, :]])
                attachment_input = attachment_input.unsqueeze(0).expand(state.word_position, attachment_input.shape[0])
                #print(attachment_input.shape, attachment_embeddings.shape)
                output_hx = torch.cat([attachment_input, attachment_embeddings], axis=1)
                for output_layer in self.output_layers:
                    output_hx = self.nonlinearity(output_hx)
                    output_hx = output_layer(output_hx)
                left_output = self.output_left(self.nonlinearity(output_hx))
                left_deprel = self.output_deprel(self.nonlinearity(output_hx))

                # truncate the outputs to only be the current heads,
                # then judge the right attachments
                current_heads = torch.tensor(state.current_heads[:-1], dtype=torch.long)
                output_hx = output_hx[current_heads, :]
                right_output = self.output_right(self.nonlinearity(output_hx))
                right_deprel = self.output_deprel(self.nonlinearity(output_hx))
                final_output[state_idx] = [final_output[state_idx][0], left_output.squeeze(1), right_output.squeeze(1)]
            left_deprels.append(left_deprel)
            right_deprels.append(right_deprel)
        final_output = [torch.cat(x) for x in final_output]
        return final_output, left_deprels, right_deprels, transition_h0, transition_c0, partial_tree_h0, partial_tree_c0

    def update_subtree_embeddings(self, states, transitions):
        embeddings = []
        if self.args['transition_subtree_combination'] == SubtreeCombination.NONE:
            for state, transition in zip(states, transitions):
                if isinstance(transition, (ProjectiveRight, NonprojectiveRight)):
                    embeddings.append(state.subtree_embeddings[state.current_heads[-1]])
                elif isinstance(transition, ProjectiveLeft):
                    embeddings.append(state.subtree_embeddings[state.current_heads[-2]])
                elif isinstance(transition, NonprojectiveLeft):
                    embeddings.append(state.subtree_embeddings[transition.word_idx])
                else:
                    continue
            if len(embeddings) == 0:
                return states
        elif self.args['transition_subtree_combination'] in (SubtreeCombination.HEAD_LINEAR, SubtreeCombination.LINEAR, SubtreeCombination.UNTIED_LINEAR):
            head_first = self.args['transition_subtree_combination'] is not SubtreeCombination.LINEAR
            relations = []
            for state, transition in zip(states, transitions):
                if isinstance(transition, (ProjectiveRight, ProjectiveLeft)):
                    relations.append(transition.deprel)
                    left = state.subtree_embeddings[state.current_heads[-2]]
                    right = state.subtree_embeddings[state.current_heads[-1]]
                    if head_first and isinstance(transition, ProjectiveRight):
                        embeddings.append(torch.cat([right, left]))
                    else:
                        embeddings.append(torch.cat([left, right]))
                elif isinstance(transition, (NonprojectiveRight, NonprojectiveLeft)):
                    relations.append(transition.deprel)
                    left = state.subtree_embeddings[transition.word_idx]
                    right = state.subtree_embeddings[state.current_heads[-1]]
                    if head_first and isinstance(transition, NonprojectiveRight):
                        embeddings.append(torch.cat([right, left]))
                    else:
                        embeddings.append(torch.cat([left, right]))
                else:
                    continue
            if len(embeddings) == 0:
                return states
            stacked_embeddings = torch.stack(embeddings, dim=0)
            # the initial attempt was a single merge matrix
            # without the /2, effectively doubling the magnitude of the inputs led to the embeddings blowing up
            if self.args['transition_subtree_combination'] is SubtreeCombination.LINEAR:
                stacked_embeddings = self.transition_subtree_nonlinearity(stacked_embeddings)
                embeddings = self.merge_subtrees(stacked_embeddings) / 2
            elif self.args['transition_subtree_combination'] is SubtreeCombination.HEAD_LINEAR:
                heads = stacked_embeddings[:, :self.args['hidden_dim'] * 2]
                stacked_embeddings = self.transition_subtree_nonlinearity(stacked_embeddings)
                embeddings = self.merge_subtrees(stacked_embeddings) / 10 + heads * 0.8
            elif self.args['transition_subtree_combination'] is SubtreeCombination.UNTIED_LINEAR:
                heads = stacked_embeddings[:, :self.args['hidden_dim'] * 2]
                stacked_embeddings = self.transition_subtree_nonlinearity(stacked_embeddings)
                embeddings = [self.merge_subtrees[deprel](emb) / 10 + head * 0.8 for deprel, emb, head in zip(relations, stacked_embeddings, heads)]
        elif self.args['transition_subtree_combination'] == SubtreeCombination.BILINEAR:
            # TODO: this one explodes almost immediately
            left = []
            right = []
            for state, transition in zip(states, transitions):
                if isinstance(transition, (ProjectiveRight, ProjectiveLeft)):
                    left.append(state.subtree_embeddings[state.current_heads[-2]])
                    right.append(state.subtree_embeddings[state.current_heads[-1]])
                elif isinstance(transition, (NonprojectiveRight, NonprojectiveLeft)):
                    left.append(state.subtree_embeddings[transition.word_idx])
                    right.append(state.subtree_embeddings[state.current_heads[-1]])
                else:
                    continue
            if len(left) == 0:
                return states
            left = torch.stack(left, dim=0)
            right = torch.stack(right, dim=0)
            embeddings = self.merge_subtrees(left, right) / 2
        elif self.args['transition_subtree_combination'] == SubtreeCombination.MAX:
            for state, transition in zip(states, transitions):
                if isinstance(transition, (ProjectiveRight, ProjectiveLeft)):
                    left = state.subtree_embeddings[state.current_heads[-2]]
                    right = state.subtree_embeddings[state.current_heads[-1]]
                elif isinstance(transition, (NonprojectiveRight, NonprojectiveLeft)):
                    left = state.subtree_embeddings[transition.word_idx]
                    right = state.subtree_embeddings[state.current_heads[-1]]
                else:
                    continue
                stacked = torch.stack([left, right], dim=1)
                embeddings.append(torch.max(stacked, 1).values)
            if len(embeddings) == 0:
                return states
            stacked_embeddings = torch.stack(embeddings, dim=0)
            embeddings = self.reduce_linear(stacked_embeddings)
        elif self.args['transition_subtree_combination'] in (SubtreeCombination.LSTM, SubtreeCombination.BILSTM):
            device = next(self.parameters()).device
            pieces = []
            heads = []
            for state, transition in zip(states, transitions):
                if isinstance(transition, ProjectiveLeft):
                    first = state.subtree_embeddings[state.current_heads[-2]]
                    second = state.subtree_embeddings[state.current_heads[-1]]
                elif isinstance(transition, ProjectiveRight):
                    first = state.subtree_embeddings[state.current_heads[-1]]
                    second = state.subtree_embeddings[state.current_heads[-2]]
                elif isinstance(transition, NonprojectiveLeft):
                    first = state.subtree_embeddings[transition.word_idx]
                    second = state.subtree_embeddings[state.current_heads[-1]]
                elif isinstance(transition, NonprojectiveRight):
                    first = state.subtree_embeddings[state.current_heads[-1]]
                    second = state.subtree_embeddings[transition.word_idx]
                else:
                    continue
                relation_id = torch.tensor(self.relation_to_id[transition.deprel], requires_grad=False, dtype=torch.long, device=device)
                relation_emb = self.reduce_relation_embedding(relation_id)
                heads.append(first)
                if self.args['transition_subtree_combination'] is SubtreeCombination.LSTM:
                    pieces.append((relation_emb, second, first))
                else:
                    pieces.append((relation_emb, first, second, relation_emb))
            if len(pieces) == 0:
                return states
            pieces = [torch.stack(piece, dim=0) for piece in pieces]
            lstm_input = torch.stack(pieces, dim=1)
            embeddings, _ = self.reduce_lstm(lstm_input)
            if self.args['transition_subtree_combination'] is SubtreeCombination.LSTM:
                embeddings = embeddings[-1, :, :]
            else:
                emb_forward = embeddings[2, :, :self.args['hidden_dim']]   # use the embedding for rel, first, second
                emb_reverse = embeddings[1, :, self.args['hidden_dim']:]   # use the embedding for rel, second, first
                embeddings = torch.cat([emb_forward, emb_reverse], dim=1)
            heads = torch.stack(heads, dim=0)
            embeddings = embeddings * 0.2 + heads * 0.8
        else:
            raise ValueError("Unknown transition_subtree_combination %s" % self.args['transition_subtree_combination'])
        embedding_idx = 0
        # TODO: when NonprojectiveLeft is below the top head of the subtree
        # it is in, we should propagate the changes up the subtree
        for state, transition in zip(states, transitions):
            if isinstance(transition, (ProjectiveRight, NonprojectiveRight)):
                state.subtree_embeddings[state.current_heads[-1]] = embeddings[embedding_idx]
                embedding_idx += 1
            elif isinstance(transition, ProjectiveLeft):
                state.subtree_embeddings[state.current_heads[-2]] = embeddings[embedding_idx]
                embedding_idx += 1
            elif isinstance(transition, NonprojectiveLeft):
                state.subtree_embeddings[transition.word_idx] = embeddings[embedding_idx]
                embedding_idx += 1
        return states

    def update_partial_tree_lstm(self, states, state_idxs, partial_tree_h0, partial_tree_c0):
        new_states = []
        new_state_idxs = []
        for state_idx, state in zip(state_idxs, states):
            # TODO: if we switch to a partial tree representation, this may not be true any more
            # (eg, if creating a partial tree updates the word embedding to a tree embedding)
            # in which case we would need to update the h0/c0 and convert the word embedding when doing the Shift
            if isinstance(state.transitions[-1], Shift):
                state.subtree_lstm_embeddings.append(SubtreeLSTMEmbedding(partial_tree_h0[:, state_idx, :], partial_tree_c0[:, state_idx, :]))
                state.subtree_embeddings[state.current_heads[-1]] = state.word_embeddings[state.current_heads[-1]]
            elif isinstance(state.transitions[-1], Finalize):
                continue
            else:
                new_states.append(state)
                new_state_idxs.append(state_idx)
        states = new_states
        state_idxs = new_state_idxs
        while len(states) > 0:
            partial_tree_embeddings = []
            partial_tree_h0 = []
            partial_tree_c0 = []
            for state_idx, state in zip(state_idxs, states):
                if len(state.subtree_lstm_embeddings) == 0:
                    partial_tree_embeddings.append(self.partial_tree_start)
                    partial_tree_h0.append(self.partial_tree_h0)
                    partial_tree_c0.append(self.partial_tree_c0)
                else:
                    head = len(state.subtree_lstm_embeddings) - 1
                    head = state.current_heads[head]
                    partial_tree_embeddings.append(state.subtree_embeddings[head])
                    partial_tree_h0.append(state.subtree_lstm_embeddings[-1].h0)
                    partial_tree_c0.append(state.subtree_lstm_embeddings[-1].c0)
            partial_tree_embeddings = torch.stack(partial_tree_embeddings, dim=0).unsqueeze(0)
            partial_tree_h0 = torch.stack(partial_tree_h0, dim=1)
            partial_tree_c0 = torch.stack(partial_tree_c0, dim=1)
            partial_tree_embeddings, (partial_tree_h0, partial_tree_c0) = self.partial_tree_lstm(partial_tree_embeddings, (partial_tree_h0, partial_tree_c0))
            for state_idx, state in enumerate(states):
                # iterate over an enumerate because now we are looking at a shortened output from the LSTM
                state.subtree_lstm_embeddings.append(SubtreeLSTMEmbedding(partial_tree_h0[:, state_idx, :], partial_tree_c0[:, state_idx, :]))
            new_states = []
            new_state_idxs = []
            for state_idx, state in zip(state_idxs, states):
                if len(state.subtree_lstm_embeddings) < len(state.current_heads):
                    new_states.append(state)
                    new_state_idxs.append(state_idx)
            states = new_states
            state_idxs = new_state_idxs


    def loss(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        # lstm_outputs will be a list of tensors for each sentence
        #   max(len) x args['hidden_dim']*2
        lstm_outputs = self.embed(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        states = self.build_initial_states(head, deprel, text, lstm_outputs, sentlens)
        device = next(self.parameters()).device

        total_loss = 0

        iteration = 0
        while len(states) > 0:
            iteration += 1
            #print("ITERATION %d" % iteration)
            output_hx, left_deprels, right_deprels, transition_h0, transition_c0, partial_tree_h0, partial_tree_c0 = self.forward(states)
            new_states = []
            for state_idx, state in enumerate(states):
                # one hot vectors will be made in the following order:
                # Shift - Finalize - word_position left attachments - num_heads-1 right attachments
                if len(state.current_heads) <= 1:
                    one_hot = torch.zeros(2)
                else:
                    # 2 for shift & finalize
                    # state.num_heads - 1 for right attachments
                    # state.word_position for the left attachments
                    num_hots = len(state.current_heads) + state.word_position + 1
                    one_hot = torch.zeros(num_hots)
                num_transitions = len(state.transitions)
                gold_transition = state.gold_sequence[num_transitions]
                #print(state_idx, gold_transition, len(state.current_heads), state.word_position)
                if isinstance(gold_transition, Shift):
                    one_hot[0] = 1
                elif isinstance(gold_transition, Finalize):
                    one_hot[1] = 1
                elif isinstance(gold_transition, ProjectiveLeft) or isinstance(gold_transition, NonprojectiveLeft):
                    if isinstance(gold_transition, ProjectiveLeft):
                        head = state.current_heads[-2]
                    else:
                        head = gold_transition.word_idx
                    #print("  Left", head, num_hots)
                    # words are indexed at 1
                    # so there should never be a head for word 0
                    # hence the first word needs to be at one_hot[2]
                    one_hot[head+1] = 1

                    # also, include a loss for the deprel
                    deprel = gold_transition.deprel
                    deprel_idx = self.relation_to_id[deprel]
                    deprel_one_hot = torch.zeros(len(self.relations))
                    deprel_one_hot[deprel_idx] = 1
                    deprel_one_hot = deprel_one_hot.to(device)
                    # the word_embeddings list has an entry for root at 0,
                    # whereas the words are indexed from 1
                    # here, though, we only want to attach words to previous heads
                    # so we do head-1
                    deprel_output = left_deprels[state_idx][head-1]
                    total_loss += self.deprel_loss_function(deprel_output, deprel_one_hot)
                elif isinstance(gold_transition, ProjectiveRight) or isinstance(gold_transition, NonprojectiveRight):
                    if isinstance(gold_transition, ProjectiveRight):
                        head = len(state.current_heads) - 2
                    else:
                        head = state.current_heads.index(gold_transition.word_idx)
                    hot_idx = 2+state.word_position+head
                    #print("  Right", hot_idx, num_hots)
                    one_hot[hot_idx] = 1

                    # also, include a loss for the deprel
                    deprel = gold_transition.deprel
                    deprel_idx = self.relation_to_id[deprel]
                    deprel_one_hot = torch.zeros(len(self.relations))
                    deprel_one_hot[deprel_idx] = 1
                    deprel_one_hot = deprel_one_hot.to(device)
                    deprel_output = right_deprels[state_idx][head]
                    total_loss += self.deprel_loss_function(deprel_output, deprel_one_hot)
                one_hot = one_hot.to(device)
                total_loss += self.transition_loss_function(output_hx[state_idx], one_hot)
            gold_transitions = [state.gold_sequence[len(state.transitions)] for state in states]
            states = self.update_subtree_embeddings(states, gold_transitions)
            states = [gold_transition.apply(state) for state, gold_transition in zip(states, gold_transitions)]
            for state_idx, state in enumerate(states):
                # TODO: can this be moved into .apply()
                state.transition_lstm_embeddings.append(TransitionLSTMEmbedding(transition_h0[:, state_idx, :], transition_c0[:, state_idx, :]))
            self.update_partial_tree_lstm(states, range(len(states)), partial_tree_h0, partial_tree_c0)
            states = [state for state in states if not isinstance(state.transitions[-1], Finalize)]

        return total_loss

    def predict(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        lstm_outputs = self.embed(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        states = self.build_initial_states(head, deprel, text, lstm_outputs, sentlens)
        device = next(self.parameters()).device

        finished_states = []
        orig_state_idx = list(range(len(states)))

        iteration = 0
        while len(states) > 0:
            iteration += 1
            #print("ITERATION %d" % iteration)
            output_hx, left_deprels, right_deprels, transition_h0, transition_c0, partial_tree_h0, partial_tree_c0 = self.forward(states)
            transitions = []
            for state_idx, state in enumerate(states):
                #print(state.word_position, state.current_heads)
                def idx_to_action(idx, left_deprel, right_deprel):
                    if idx == 0:
                        return Shift()
                    if idx == 1:
                        return Finalize()
                    if idx < state.word_position + 2:
                        # again, head of 0 is not possible for a left transition
                        # so we start counting as if the first left index (idx==2) represents 1
                        head = idx - 1
                        # ... but we need to index this by -1 extra
                        max_deprel = torch.argmax(left_deprel[head-1]).item()
                        deprel = self.relations[max_deprel]
                        if head == state.current_heads[-2]:
                            return ProjectiveLeft(deprel=deprel)
                        return NonprojectiveLeft(deprel=deprel, word_idx=head)
                    if idx < state.word_position + len(state.current_heads) + 1:
                        head = idx - 2 - state.word_position
                        max_deprel = torch.argmax(right_deprel[head]).item()
                        deprel = self.relations[max_deprel]
                        if head == len(state.current_heads) - 2:
                            return ProjectiveRight(deprel=deprel)
                        return NonprojectiveRight(deprel=deprel, word_idx=state.current_heads[head])
                    raise AssertionError("Prediction idx was outside the expected number of transitions")
                _, indices = output_hx[state_idx].sort(descending=True)
                for idx in indices:
                    action = idx_to_action(idx.item(), left_deprels[state_idx], right_deprels[state_idx])
                    if action.is_legal(state):
                        transitions.append(action)
                        break
                else:
                    # no actions were legal?  this is a serious problem
                    raise AssertionError("Found a state with no legal actions!")
            #print(transitions[0])
            #print(len(states[0].subtree_lstm_embeddings),
            #      len(states[0].current_heads), states[0].current_heads)
            #if len(states[0].subtree_lstm_embeddings) > 0:
            #    print(torch.linalg.norm(states[0].subtree_lstm_embeddings[-1].h0), torch.linalg.norm(states[0].subtree_lstm_embeddings[-1].c0))
            states = self.update_subtree_embeddings(states, transitions)
            states = [transition.apply(state) for state, transition in zip(states, transitions)]
            #print(len(states[0].subtree_lstm_embeddings),
            #      len(states[0].current_heads), states[0].current_heads)
            #if len(states[0].subtree_lstm_embeddings) > 0:
            #    print(torch.linalg.norm(states[0].subtree_lstm_embeddings[-1].h0), torch.linalg.norm(states[0].subtree_lstm_embeddings[-1].c0))
            for state_idx, state in enumerate(states):
                # TODO: can this be moved into .apply()
                state.transition_lstm_embeddings.append(TransitionLSTMEmbedding(transition_h0[:, state_idx, :], transition_c0[:, state_idx, :]))
            self.update_partial_tree_lstm(states, range(len(states)), partial_tree_h0, partial_tree_c0)
            #print(len(states[0].subtree_lstm_embeddings),
            #      len(states[0].current_heads), states[0].current_heads)
            #if len(states[0].subtree_lstm_embeddings) > 0:
            #    print(torch.linalg.norm(states[0].subtree_lstm_embeddings[-1].h0), torch.linalg.norm(states[0].subtree_lstm_embeddings[-1].c0))
            new_states = []
            new_state_idx = []
            for state_idx, (state, transition, orig_idx) in enumerate(zip(states, transitions, orig_state_idx)):
                if isinstance(transition, Finalize):
                    finished_states.append((state, orig_idx))
                else:
                    new_states.append(state)
                    new_state_idx.append(orig_idx)
            states = new_states
            orig_state_idx = new_state_idx
        states = [x[0] for x in finished_states]
        orig_idx = [x[1] for x in finished_states]
        states = unsort(states, orig_idx)
        predictions = []
        for state in states:
            state_predictions = []
            for word_idx in range(1, state.num_words+1):
                head = next(state.parsed_graph.successors(word_idx))
                deprel = state.parsed_graph.get_edge_data(word_idx, head)['deprel']
                state_predictions.append((head, deprel))
            predictions.append(state_predictions)
        return predictions

    def build_initial_states(self, head, deprel, text, lstm_outputs, sentlens):
        if self.training:
            sentlens = [x-1 for x in sentlens]
            deprel = [self.vocab['deprel'].unmap(deps) for deps in deprel]
            states = states_from_heads(head, deprel, text, sentlens)
        else:
            states = [state_from_text(sentence) for sentence in text]
        updated_states = []
        # TODO: list comprehension?
        for state, lstm_output, sentlen in zip(states, lstm_outputs, sentlens):
            # the sentences are all prepended with root
            # which is fine, since we need an embedding for word 0
            state = state._replace(word_embeddings=lstm_output,
                                   subtree_embeddings={})
            updated_states.append(state)
        return updated_states
