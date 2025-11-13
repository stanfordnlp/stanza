import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from stanza.models.common.utils import unsort
from stanza.models.common.vocab import VOCAB_PREFIX_SIZE
from stanza.models.depparse.model import BaseParser
from stanza.models.depparse.transition.state import state_from_text, states_from_heads, TransitionLSTMEmbedding
from stanza.models.depparse.transition.transitions import Shift, Finalize, ProjectiveLeft, ProjectiveRight, NonprojectiveLeft, NonprojectiveRight

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

        # the bidirectional LSTM is x2, adding in the partial trees is another x1
        self.final_hidden_dim = self.transition_hidden_dim + self.args['hidden_dim'] * 3
        self.output_layers = nn.ModuleList([nn.Linear(self.final_hidden_dim, self.final_hidden_dim)])

        self.nonlinearity = nn.ReLU()
        self.output_basic = nn.Linear(self.final_hidden_dim, 2)
        self.output_left = nn.Linear(self.final_hidden_dim, 1)
        self.output_right = nn.Linear(self.final_hidden_dim, 1)
        # this will be used to predict the relation if a transition is chosen
        self.output_deprel = nn.Linear(self.final_hidden_dim, len(self.relations))

        # TODO: maybe make an attention layer?
        # maybe split this across different relations or right/left?
        self.merge_words = nn.Linear(self.args['hidden_dim'] * 4, self.args['hidden_dim'] * 2)

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

        partial_trees = []
        for state in states:
            state_partial_trees = [self.partial_tree_start]
            for head in state.current_heads:
                state_partial_trees.append(state.word_embeddings[head-1])
            state_partial_trees = torch.stack(state_partial_trees)
            partial_trees.append(state_partial_trees)
        packed_partial_trees = torch.nn.utils.rnn.pack_sequence(partial_trees, enforce_sorted=False)
        partial_tree_output, _ = self.partial_tree_lstm(packed_partial_trees)
        partial_tree_output, partial_tree_lens = pad_packed_sequence(partial_tree_output)
        partial_tree_lens = partial_tree_lens - 1
        # TODO: isn't this 'gather' or something like that?
        partial_tree_embeddings = torch.zeros_like(partial_tree_output[0, :, :])
        for idx, length in enumerate(partial_tree_lens):
            partial_tree_embeddings[idx, :] = partial_tree_output[length, idx, :]

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
                # TODO: use a tensor to combine words into tree embeddings
                attachment_embeddings = [torch.cat([state.word_embeddings[x], state.word_embeddings[state.current_heads[-1]]])
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
        return final_output, left_deprels, right_deprels, transition_h0, transition_c0

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
            output_hx, left_deprels, right_deprels, transition_h0, transition_c0 = self.forward(states)
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
                # TODO: would need to include a callback for combining words, if we use a subtree combination embedding
                state = gold_transition.apply(state)
                if not isinstance(gold_transition, Finalize):
                    # TODO: can this be moved into .apply()
                    state.transition_lstm_embeddings.append(TransitionLSTMEmbedding(transition_h0[:, state_idx, :], transition_c0[:, state_idx, :]))
                    new_states.append(state)
            states = new_states

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
            output_hx, left_deprels, right_deprels, transition_h0, transition_c0 = self.forward(states)
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
            states = [transition.apply(state) for state, transition in zip(states, transitions)]
            for state_idx, state in enumerate(states):
                # TODO: can this be moved into .apply()
                state.transition_lstm_embeddings.append(TransitionLSTMEmbedding(transition_h0[:, state_idx, :], transition_c0[:, state_idx, :]))
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
            state = state._replace(word_embeddings=lstm_output)
            updated_states.append(state)
        return updated_states
