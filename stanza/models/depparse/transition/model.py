from collections import Counter, namedtuple
from enum import Enum

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stanza.models.common.utils import build_nonlinearity, unsort
from stanza.models.common.vocab import VOCAB_PREFIX_SIZE
from stanza.models.depparse.model import BaseParser, EmbeddingParser
from stanza.models.depparse.transition import dynamic_oracle
from stanza.models.depparse.transition.state import state_from_text, states_from_data_batch, TransitionLSTMEmbedding, SubtreeLSTMEmbedding
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

class BatchStats(namedtuple("BatchStats", ['batch_loss', 'transitions_correct', 'transitions_incorrect', 'repairs_used'])):
    @staticmethod
    def empty():
        return BatchStats(0.0, 0, 0, Counter())

    def __add__(self, other):
        transitions_correct = self.transitions_correct + other.transitions_correct
        transitions_incorrect = self.transitions_incorrect + other.transitions_incorrect
        repairs_used = self.repairs_used + other.repairs_used
        batch_loss = self.batch_loss + other.batch_loss
        return BatchStats(batch_loss, transitions_correct, transitions_incorrect, repairs_used)

    def __str__(self):
        stats = "Loss: %f\nT Correct: %d\nT Incorrect: %d" % (self.batch_loss, self.transitions_correct, self.transitions_incorrect)
        if len(self.repairs_used) > 0:
            repairs = "Oracle repairs:\n  %s" % "\n  ".join("%s (%s): %d" % (x.name, x.value, y) for x, y in self.repairs_used.most_common())
            stats = "%s\n%s" % (stats, repairs)
        return stats

class TransitionParser(EmbeddingParser):
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

        self.nonlinearity = nn.ReLU()
        self.transition_subtree_nonlinearity = build_nonlinearity(self.args.get('transition_subtree_nonlinearity'))
        self.drop = nn.Dropout(self.args['dropout'])

        # the bidirectional LSTM is x2, adding in the partial trees is another x1
        self.word_hidden_dim = self.transition_hidden_dim + self.args['hidden_dim'] * 3
        self.word_output_layers = nn.Sequential(self.nonlinearity,
                                                self.drop,
                                                nn.Linear(self.word_hidden_dim, self.word_hidden_dim),
                                                self.nonlinearity,
                                                self.drop,
                                                nn.Linear(self.word_hidden_dim, self.word_hidden_dim))
        self.merge_words_output_dim = self.args['transition_merge_words_output_dim']
        self.merge_hidden_dim = self.transition_hidden_dim + self.args['hidden_dim'] + self.merge_words_output_dim
        # Splitting this into a left and right version is close,
        # but seems to be somewhat more accurate than one layer
        #  5 model dev avg LAS  baseline  merge-two-sides
        # de_gsd                 88.75     88.93
        # en_ewt                 93.40     93.41
        # fi_tdt                 92.65     92.75
        # it_vit                 89.99     90.03
        # ta_ttb                 72.21     72.05
        # zh-hans_gsdsimp        85.17     85.20
        #
        #  5 model test avg LAS baseline  merge-two-sides
        # de_gsd                 86.31     86.60
        # en_ewt                 93.30     93.29
        # fi_tdt                 92.76     92.86
        # it_vit                 90.15     90.29
        # ta_ttb                 68.77     68.90
        # zh-hans_gsdsimp        85.27     85.48
        self.merge_output_left = nn.Sequential(self.nonlinearity,
                                               self.drop,
                                               nn.Linear(self.merge_hidden_dim, self.merge_hidden_dim),
                                               self.nonlinearity,
                                               self.drop,
                                               nn.Linear(self.merge_hidden_dim, self.merge_hidden_dim))
        self.merge_output_right = nn.Sequential(self.nonlinearity,
                                                self.drop,
                                                nn.Linear(self.merge_hidden_dim, self.merge_hidden_dim),
                                                self.nonlinearity,
                                                self.drop,
                                                nn.Linear(self.merge_hidden_dim, self.merge_hidden_dim))

        self.output_basic = nn.Linear(self.word_hidden_dim, 2)
        self.output_left_transition = nn.Linear(self.merge_hidden_dim, 1)
        self.output_right_transition = nn.Linear(self.merge_hidden_dim, 1)
        # this will be used to predict the relation if a transition is chosen
        self.output_left_deprel = nn.Linear(self.merge_hidden_dim, len(self.relations))
        self.output_right_deprel = nn.Linear(self.merge_hidden_dim, len(self.relations))

        # Previously we used one merge_words layer for both the right and left
        # Splitting it into two pieces makes a noticeable difference in accuracy
        # Splitting output_deprel into output_left_reprel and output_right_deprel
        #   also made a slight difference
        # On an experiment using transformers, on UD 2.17, repeated 5x
        # for averaging, using the adadelta optimizer but not the
        # second pass, we had
        #         combined   split_merge   split_relations
        #  de_gsd  88.09         88.33         88.41
        #  en_ewt  92.55         93.06         93.08
        #  it_vit  89.38         89.53         89.68
        # The first experiment with using a Bilinear layer instead of a Linear
        # greatly hurt scores and was much slower.  Perhaps it can be redone better
        #
        # Another experiment we tried was to greatly expand the width
        # of the merge_words layers, then use an attention-like softmax
        # to select which part of the wider output to use.
        # The first experiment with this wound up also being slower
        # and less effective.
        self.merge_words_right = nn.Linear(self.args['hidden_dim'] * 4, self.merge_words_output_dim)
        self.merge_words_left = nn.Linear(self.args['hidden_dim'] * 4, self.merge_words_output_dim)

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
            self.reduce_lstm = nn.LSTM(input_size=self.args['hidden_dim'] * 2, hidden_size=self.args['hidden_dim'] * 2, num_layers=self.args['num_layers'], dropout=self.args['dropout'], bidirectional=True)
            self.reduce_relation_embedding = nn.Embedding(num_embeddings = len(self.relations),
                                                          embedding_dim = self.args['hidden_dim'] * 2)
        else:
            raise ValueError("Unknown transition_subtree_combination %s" % self.args['transition_subtree_combination'])

        self.transition_loss_function = nn.CrossEntropyLoss(reduction='sum')
        self.deprel_loss_function = nn.CrossEntropyLoss(reduction='sum')

    def empty_stats(self):
        return BatchStats.empty()

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
        output_hx = self.word_output_layers(output_hx)
        # batch size x 2 - Shift or Finalize
        basic_output = self.output_basic(self.drop(self.nonlinearity(output_hx)))
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
                attachment_embeddings = torch.stack(attachment_embeddings, dim=0)
                attachment_embeddings_left = self.merge_words_left(attachment_embeddings)

                # in addition to the current words, we also use the current transition and partial tree
                # LSTM outputs to determine the scores of each attachment (and possible dependency)
                attachment_input = torch.cat([transition_embeddings[state_idx, :], partial_tree_embeddings[state_idx, :]])
                attachment_input_left = attachment_input.expand(state.word_position, attachment_input.shape[0])
                left_arc_hx = torch.cat([attachment_input_left, attachment_embeddings_left], axis=1)
                left_arc_hx = self.merge_output_left(left_arc_hx)
                left_output = self.output_left_transition(self.drop(self.nonlinearity(left_arc_hx)))
                left_deprel = self.output_left_deprel(self.drop(self.nonlinearity(left_arc_hx)))

                # truncate the outputs to only be the current heads,
                # then judge the right attachments
                current_heads = torch.tensor(state.current_heads[:-1], dtype=torch.long)
                attachment_embeddings_right = attachment_embeddings[current_heads, :]
                attachment_embeddings_right = self.merge_words_right(attachment_embeddings_right)
                attachment_input_right = attachment_input.unsqueeze(0).expand(current_heads.shape[0], attachment_input.shape[0])
                right_arc_hx = torch.cat([attachment_input_right, attachment_embeddings_right], axis=1)
                right_arc_hx = self.merge_output_right(right_arc_hx)
                right_output = self.output_right_transition(self.drop(self.nonlinearity(right_arc_hx)))
                right_deprel = self.output_right_deprel(self.drop(self.nonlinearity(right_arc_hx)))
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
                    head = state.current_heads[-2]
                    child = state.current_heads[-1]
                elif isinstance(transition, ProjectiveRight):
                    head = state.current_heads[-1]
                    child = state.current_heads[-2]
                elif isinstance(transition, NonprojectiveLeft):
                    head = transition.word_idx
                    child = state.current_heads[-1]
                elif isinstance(transition, NonprojectiveRight):
                    head = state.current_heads[-1]
                    child = transition.word_idx
                else:
                    continue
                children = [child]
                if head in state.parsed_graph:
                    children.extend([x for x in state.parsed_graph.predecessors(head)])
                children.sort()
                children = [state.word_embeddings[child] for child in children]
                # TODO: in some way incorporate the relation used for the children?
                head_emb = state.word_embeddings[head]
                heads.append(head_emb)
                pieces.append([head_emb] + children + [head_emb])
            if len(pieces) == 0:
                return states
            piece_lens = [len(x) for x in pieces]
            max_len = max(piece_lens)
            pieces = [torch.stack(piece, dim=0) for piece in pieces]
            lstm_input = torch.zeros((max_len, len(pieces), pieces[0].shape[1]), dtype=pieces[0].dtype, device=device)
            for piece_idx, piece in enumerate(pieces):
                lstm_input[:len(piece), piece_idx, :] = piece
            lstm_input = pack_padded_sequence(lstm_input, piece_lens, enforce_sorted=False)
            lstm_output, _ = self.reduce_lstm(lstm_input)
            lstm_output, _ = pad_packed_sequence(lstm_output)

            if self.args['transition_subtree_combination'] is SubtreeCombination.LSTM:
                embeddings = [lstm_output[piece_len-1, piece_idx, :] for piece_idx, piece_len in enumerate(piece_lens)]
                embeddings = torch.stack(embeddings, dim=0)
            else:
                emb_forward = [lstm_output[piece_len-1, piece_idx, :2*self.args['hidden_dim']] for piece_idx, piece_len in enumerate(piece_lens)]
                emb_forward = torch.stack(emb_forward, dim=0)
                emb_reverse = lstm_output[0, :, 2*self.args['hidden_dim']:]
                embeddings = emb_forward + emb_reverse
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

    def calculate_iteration_loss(self, states, gold_transitions, output_hx, left_deprels, right_deprels):
        device = next(self.parameters()).device
        total_loss = 0.0
        deprel_one_hots = []
        deprel_hx = []
        for state_idx, (state, gold_transition) in enumerate(zip(states, gold_transitions)):
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
                deprel_hx.append(deprel_output)
                deprel_one_hots.append(deprel_one_hot)
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
                deprel_hx.append(deprel_output)
                deprel_one_hots.append(deprel_one_hot)
            one_hot = one_hot.to(device)
            total_loss += self.transition_loss_function(output_hx[state_idx], one_hot)
        if len(deprel_one_hots) > 0:
            deprel_one_hots = torch.stack(deprel_one_hots, dim=0)
            deprel_hx = torch.stack(deprel_hx, dim=0)
            total_loss += self.deprel_loss_function(deprel_hx, deprel_one_hots) * self.args['transition_relation_learning_factor']
        return total_loss

    def loss(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        # lstm_outputs will be a list of tensors for each sentence
        #   max(len) x args['hidden_dim']*2
        lstm_outputs = self.embed(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        states = self.build_initial_states(head, deprel, text, lstm_outputs, sentlens)

        transitions_correct = 0
        transitions_incorrect = 0
        repairs_used = Counter()

        total_loss = 0
        iteration = 0
        while len(states) > 0:
            iteration += 1
            #print("ITERATION %d" % iteration)
            output_hx, left_deprels, right_deprels, transition_h0, transition_c0, partial_tree_h0, partial_tree_c0 = self.forward(states)
            gold_transitions = [state.gold_sequence[len(state.transitions)] for state in states]

            chosen_transitions = self.choose_transitions(self.relations, states, output_hx, left_deprels, right_deprels)
            transitions_correct += sum(x == y for x, y in zip(gold_transitions, chosen_transitions))
            transitions_incorrect += sum(x != y for x, y in zip(gold_transitions, chosen_transitions))

            iteration_loss = self.calculate_iteration_loss(states, gold_transitions, output_hx, left_deprels, right_deprels)
            total_loss += iteration_loss

            transitions = []
            new_states = []
            for state, gold_transition, chosen_transition in zip(states, gold_transitions, chosen_transitions):
                # the learning uses the output states for training
                # the chosen transition should always be legal, though
                assert chosen_transition.is_legal(state)

                repair, new_sequence = dynamic_oracle.repair(state, gold_transition, chosen_transition)
                repairs_used[repair] += 1
                if new_sequence is not None:
                    state = state._replace(gold_sequence=new_sequence)
                    transitions.append(chosen_transition)
                else:
                    transitions.append(gold_transition)
                new_states.append(state)

            states = new_states
            states = self.update_states(states, transitions, transition_h0, transition_c0, partial_tree_h0, partial_tree_c0)
            states = [state for state in states if not isinstance(state.transitions[-1], Finalize)]

        return total_loss, BatchStats(batch_loss=total_loss.item(),
                                      transitions_correct=transitions_correct,
                                      transitions_incorrect=transitions_incorrect,
                                      repairs_used=repairs_used)

    @staticmethod
    def idx_to_action(relations, state, idx, left_deprel, right_deprel):
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
            deprel = relations[max_deprel]
            if head == state.current_heads[-2]:
                return ProjectiveLeft(deprel=deprel)
            return NonprojectiveLeft(deprel=deprel, word_idx=head)
        if idx < state.word_position + len(state.current_heads) + 1:
            head = idx - 2 - state.word_position
            max_deprel = torch.argmax(right_deprel[head]).item()
            deprel = relations[max_deprel]
            if head == len(state.current_heads) - 2:
                return ProjectiveRight(deprel=deprel)
            return NonprojectiveRight(deprel=deprel, word_idx=state.current_heads[head])
        raise AssertionError("Prediction idx was outside the expected number of transitions")

    @staticmethod
    def choose_transitions(relations, states, output_hx, left_deprels, right_deprels):
        transitions = []
        for state_idx, state in enumerate(states):
            #print(state.word_position, state.current_heads)
            _, indices = output_hx[state_idx].sort(descending=True)
            for idx in indices:
                action = TransitionParser.idx_to_action(relations, state, idx.item(), left_deprels[state_idx], right_deprels[state_idx])
                if action.is_legal(state):
                    transitions.append(action)
                    break
            else:
                # no actions were legal?  this is a serious problem
                raise AssertionError("Found a state with no legal actions!")
        return transitions

    def update_states(self, states, transitions, transition_h0, transition_c0, partial_tree_h0, partial_tree_c0):
        states = self.update_subtree_embeddings(states, transitions)
        states = [transition.apply(state) for state, transition in zip(states, transitions)]
        for state_idx, state in enumerate(states):
            # TODO: can this be moved into .apply()
            state.transition_lstm_embeddings.append(TransitionLSTMEmbedding(transition_h0[:, state_idx, :], transition_c0[:, state_idx, :]))
        self.update_partial_tree_lstm(states, range(len(states)), partial_tree_h0, partial_tree_c0)
        return states

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
            transitions = self.choose_transitions(self.relations, states, output_hx, left_deprels, right_deprels)

            states = self.update_states(states, transitions, transition_h0, transition_c0, partial_tree_h0, partial_tree_c0)

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
            states = states_from_data_batch(self.vocab['deprel'], head, deprel, text, sentlens)
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

class EnsembleTransitionParser(BaseParser):
    def __init__(self, args, vocab, models):
        super().__init__(args, vocab)
        self.models = nn.ModuleList(models)

    # TODO: refactor with EnsembleGraphParser
    def get_params(self, skip_modules):
        params = []
        args = []
        for model in self.models:
            params.append(model.get_params(skip_modules))
            config = dict(model.args)
            # sanitize enums for torch.load(weights_only=True)
            if 'transition_subtree_combination' in config:
                config['transition_subtree_combination'] = config['transition_subtree_combination'].name
            args.append(config)
        checkpoint = {
            "num_models": len(self.models),
            "params": params,
            "args": args,
        }
        return checkpoint

    def load_params(self, checkpoint):
        for model, params in zip(self.models, checkpoint["params"]):
            model.load_params(params)

    def loss(self, *args, **kwargs):
        raise NotImplementedError("Cannot train ensemble parser")


    def get_device(self):
        return self.models[0].get_device()

    def forward(self, model_states):
        model_forwards = [model.forward(states) for model, states in zip(self.models, model_states)]

        output_hx = []
        left_deprels = []
        right_deprels = []
        for state_idx in range(len(model_forwards[0][0])):
            model_output_hx = torch.stack([x[0][state_idx] for x in model_forwards], dim=0)
            output_hx.append(torch.sum(model_output_hx, dim=0))

            # TODO: would it be simpler to return a 0 dim tensor?
            if model_forwards[0][1][state_idx] is not None:
                model_left_deprels = torch.stack([x[1][state_idx] for x in model_forwards], dim=0)
                left_deprels.append(torch.sum(model_left_deprels, dim=0))
            else:
                left_deprels.append([])

            if model_forwards[0][2][state_idx] is not None:
                model_right_deprels = torch.stack([x[2][state_idx] for x in model_forwards], dim=0)
                right_deprels.append(torch.sum(model_right_deprels, dim=0))
            else:
                right_deprels.append([])

        model_transition_h0 = [x[3] for x in model_forwards]
        model_transition_c0 = [x[4] for x in model_forwards]
        model_partial_tree_h0 = [x[5] for x in model_forwards]
        model_partial_tree_c0 = [x[6] for x in model_forwards]
        return output_hx, left_deprels, right_deprels, model_transition_h0, model_transition_c0, model_partial_tree_h0, model_partial_tree_c0

    def predict(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
        device = self.get_device()

        model_states = []
        for model in self.models:
            lstm_outputs = model.embed(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
            states = model.build_initial_states(head, deprel, text, lstm_outputs, sentlens)
            model_states.append(states)

        finished_states = []
        orig_state_idx = list(range(len(sentlens)))

        iteration = 0
        while len(model_states[0]) > 0:
            iteration += 1

            # output_hx, left_deprels, and right_deprels are already collapsed into one summed value
            # the transition and partial_tree vectors are a list of M items long (M=#models)
            output_hx, left_deprels, right_deprels, model_transition_h0, model_transition_c0, model_partial_tree_h0, model_partial_tree_c0 = self.forward(model_states)

            transitions = TransitionParser.choose_transitions(self.models[0].relations, model_states[0], output_hx, left_deprels, right_deprels)

            new_model_states = []
            for model, states, transition_h0, transition_c0, partial_tree_h0, partial_tree_c0 in zip(self.models, model_states, model_transition_h0, model_transition_c0, model_partial_tree_h0, model_partial_tree_c0):
                states = model.update_subtree_embeddings(states, transitions)
                states = [transition.apply(state) for state, transition in zip(states, transitions)]
                for state_idx, state in enumerate(states):
                    state.transition_lstm_embeddings.append(TransitionLSTMEmbedding(transition_h0[:, state_idx, :], transition_c0[:, state_idx, :]))
                model.update_partial_tree_lstm(states, range(len(states)), partial_tree_h0, partial_tree_c0)
                new_model_states.append(states)
            model_states = new_model_states

            # currently we only keep the results from one state - we
            # don't pay attention to the outputs further downstream,
            # after all
            new_state_idx = []
            for state_idx, (state, transition, orig_idx) in enumerate(zip(model_states[0], transitions, orig_state_idx)):
                if isinstance(transition, Finalize):
                    finished_states.append((state, orig_idx))
                else:
                    new_state_idx.append(orig_idx)
            orig_state_idx = new_state_idx

            new_model_states = []
            for model, states in zip(self.models, model_states):
                new_states = []
                for state_idx, (state, transition) in enumerate(zip(states, transitions)):
                    if not isinstance(transition, Finalize):
                        new_states.append(state)
                new_model_states.append(new_states)
            model_states = new_model_states

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

