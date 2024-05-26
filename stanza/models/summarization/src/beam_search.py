"""
Run beam search decoding from a trained abstractive summarization model
"""
import torch

from typing import List, Tuple, Mapping, Any
from stanza.models.summarization.src.model import BaselineSeq2Seq
from stanza.models.common.vocab import BaseVocab, UNK

class Hypothesis():

    """
    Represents a hypothesis during beam search. Holds all information needed for the hypothesis.
    """

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
        """
        Args:
            tokens (List[int]): The ids of tokens that form summary so far.
            log_probs (List[float]): List of the log probabilities of the tokens so far. (same length as tokens)
            state (Tuple[Tensor, Tensor]): Current state of the decoder LSTM, tuple of the LSTM hidden + cell state.
            attn_dists (List[Tensor]): List of the attention distributions at each point of the decoder (same length as tokens)
            p_gens (List[float]): Values of the generation probabilities (same length as tokens). None if not using pointer-gen.
            coverage (Tensor): Current coverage vector. None if not using coverage.
        """ 
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state 
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        """
        Return a new hypothesis, extended with the information from the latest step of beam search.

        Args:
            token (int): The latest token ID produced by beam search
            log_prob (float): Log probability of the latest token
            state (Tuple[Tensor, Tensor]): Current decoder state of the hidden and cell state.
            attn_dist (Tensor): Current attention distribution from latest step.
            p_gen (float): Generation probability from latest step.
            coverage (Tensor): Latest coverage vector, or None if not using coverage.
        
        Returns:
            Hypothesis() : New hypothesis for next step
        """
        new_hypothesis = Hypothesis(
                                    tokens=self.tokens + [token],
                                    log_probs=self.log_probs + [log_prob],
                                    state=state,
                                    attn_dists = self.attn_dists + [attn_dist], 
                                    p_gens=self.p_gens + [p_gen],
                                    coverage=coverage
                                    )
        return new_hypothesis

    def get_latest_token(self):
        # Get the last token decoded in this hypothesis
        return self.tokens[-1]

    def get_log_prob(self):
        # The sum of the log probabilities so far
        return sum(self.log_probs) 

    def get_avg_log_prob(self):
        # Normalize by sequence length (longer sequences will always have lower probability)
        return self.get_log_prob / len(self.tokens)


def run_beam_search(model: BaselineSeq2Seq, vocab: BaseVocab, example: List[str], beam_size: int,
                    max_dec_steps: int, min_dec_steps: int):
    """
    Performs beam search decoding on an example.

    Returns the hypothesis for each example with the highest average log probability.
    """
    START_TOKEN = "<s>"  # TODO find what this should be
    STOP_TOKEN = "</s>"
    batch = [example for _ in range(beam_size)]
    device = next(model.parameters()).device

    # Run encoder over the batch of examples to get the encoder hidden states and decoder init state
    # note that the batch is the same example repeated 
    enc_states, dec_hidden_init, dec_cell_init = model.run_encoder(batch)

    # enc states shape (batch size, seq len, 2 * enc hidden dim)
    # dec states are shape (batch size, dec hidden dim)
    # note that we only have one example, so the batch size should be 1

    # Initialize N-Hypotheses for beam search 
    hyps = [
        Hypothesis(
            tokens=[vocab.unit2d(START_TOKEN)],  # TODO verify that this is not unknown to the glove vocab
            log_probs=[0.0],
            state=(dec_hidden_init[0], dec_cell_init[0]),  # only one example, so get the state for that example
            p_gens=[], 
            coverage=torch.zeros(enc_states.shape[1])  # sequence length
        ) for _ in range(beam_size)
    ]
    results = []  # stores our finished hypotheses (decoded out the STOP token)

    # Run the loop while we still have decoding steps and the number of finished results is less than the beam size
    steps = 0
    while steps < max_dec_steps and len(results) < beam_size:

        latest_tokens = [h.get_latest_token() for h in hyps]  # get latest token from each hypothesis 
        latest_tokens = [t if t in range(len(vocab)) else vocab.unit2id(UNK) for t in latest_tokens]  # change any OOV words to UNK
        latest_tokens = [[vocab.id2unit(t)] for t in latest_tokens]  # convert back to word because model.decode_onestep() expects string
        hidden_states = [h.state[0] for h in hyps]
        cell_states = [h.state[1] for h in hyps]
        prev_coverage = [h.coverage for h in hyps]

        # run the decoder for one timestep, decoding out choices for the next token of each sequence
        topk_ids, topk_log_probs, new_hiddens, new_cells, attn_dists, p_gens, new_coverage = model.decode_onestep(
            examples = batch,
            latest_tokens=latest_tokens,
            enc_states=enc_states, 
            dec_hidden=torch.tensor(hidden_states, device=device),
            dec_cell=torch.tensor(cell_states, device=device),
            prev_coverage=torch.tensor(prev_coverage, device=device)
        )

        # extend current hypotheses with the possible next tokens. We determine the choices to be 2 x beam size for the choices
        all_hyps = []
        num_original_hyps = 1 if steps == 0 else len(hyps)
        for i in range(num_original_hyps):
            h, new_hidden, new_cell, attn_dist, p_gen, new_coverage_i = hyps[i], new_hiddens[i], new_cells[i], attn_dists[i], p_gens[i], new_coverage[i]
            for j in range(2 * beam_size):  # for each of the top 2*beam_size hypotheses:
                # Extend the ith hypothesis with the jth option
                new_hyp = h.extend(
                    token=topk_ids[i, j],
                    log_prob=topk_log_probs[i, j],
                    state=(new_hidden, new_cell),
                    attn_dist=attn_dist,
                    p_gen=p_gen,
                    coverage=new_coverage_i
                )
                all_hyps.append(new_hyp)
        # Filter and collect any hypotheses that have produced the end token (or are over limit)
        hyps = []
        for h in sort_hypotheses(all_hyps):  # in order of most likely h
            if h.get_latest_token() == vocab.unit2id(STOP_TOKEN):   # if we reach the stop token
                # if the hypothesis is sufficiently long, then put in results, otherwise discard
                if steps >= min_dec_steps:
                    results.append(h)
            else:  # hasn't reached stop token, so continue to expand the hypothesis
                hyps.append(h)
            if len(hyps) == beam_size or len(results) == beam_size:
                # Once we've collected beam_size-many hypotheses for the next step or beam_size-many complete hypotheses, stop
                break
        steps += 1

    # We now have either beam_size results or reached the maximum number of decoder steps 
    if len(results) == 0:
        # If we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        results = hyps 

    # Sort hypotheses by the average log probability and return the hypothesis with the highest average log prob
    hyps_sorted = sort_hypotheses(results)
    return hyps_sorted[0]


def sort_hypotheses(hyps: List[Hypothesis]):
    """
    Return of a list of Hypothesis objects sorted by descending average log prob
    """
    return sorted(hyps, key=lambda h: h.get_avg_log_prob(), reverse=True)