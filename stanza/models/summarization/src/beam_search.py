"""
Run beam search decoding from a trained abstractive summarization model
"""

class Hypothesis():

    """
    Represents a hypothesis during beam search. Holds all information needed for the hypothesis.
    """

    def __init__(self):
        pass 

    def extend(self):
        """
        Return a new hypothesis, extended with the information from the latest step of beam search.
        """

    def get_latest_token(self):
        # Get the last token decoded in this hypothesis
        pass 
    def get_log_prob(self):
        # The sum of the log probabilities so far
        pass 
    def get_avg_log_prob(self):
        # Normalize by sequence length (longer sequences will always have lower probability)
        pass


def run_beam_search():
    """
    Performs beam search decoding on a single batch of examples.

    Returns the hypothesis for each example with the highest average log probability.
    """

    # Run encoder over the batch of examples to get the encoder hidden states and decoder init state

    # Initialize N-Hypotheses for beam search 

    # Run the loop while we still have decoding steps and the number of finished results is less than the beam size

        # run the decoder for one timestep, decoding out choices for the next token of each sequence
        # TODO write this function in the model. Needs to return the top K ids, top K log probs, new hidden states,
        # attn dist, p_gens, and coverage

        # extend current hypotheses with the possible next tokens. We determine the choices to be 2 x beam size for the choices

        # Filter and collect any hypotheses that have produced the end token (or are over limit)

    # We now have either beam_size results or reached the maximum number of decoder steps 

    # Sort hypotheses by the average log probability and return the hypothesis with the highest average log prob



    pass 
