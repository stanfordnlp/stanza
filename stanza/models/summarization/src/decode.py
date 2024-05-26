"""
Takes an existing model and runs beam search decoding across many examples

"""

import torch

from typing import List, Tuple, Mapping, Any
from stanza.models.summarization.src.model import BaselineSeq2Seq
from stanza.models.summarization.src.beam_search import *
from stanza.models.common.vocab import BaseVocab, UNK
from logging import Logger

class BeamSearchDecoder():

    """
    Decoder for summarization using beam search
    """

    def __init__(self, model: BaselineSeq2Seq, vocab: BaseVocab, logger: Logger = None):
        self.model = model 
        self.vocab = vocab
        self.stop_token = "</s>"
        self.start_token = "<s>" 
        self.logger = logger
    
    def decode_examples(self, examples: List[List[str]], beam_size: int, max_dec_steps: int, min_dec_steps: int,
                        verbose: bool = False) -> List[List[str]]:
        summaries = []  # outputs 
        for i, article in enumerate(examples):
            
            try:
                # Run beam search to get the best hypothesis
                best_hyp = run_beam_search(self.model, 
                                           self.vocab, 
                                           article, 
                                           beam_size,
                                           max_dec_steps,
                                           min_dec_steps
                                           )
                
                output_ids = [int(t) for t in best_hyp.tokens[1: ]]  # exclude START tokens but not STOP because not guaranteed to contain STOP
                decoded_words = [self.vocab.id2unit(idx) for idx in output_ids]
                if self.stop_token in decoded_words:
                    fst_stop_index = decoded_words.index(self.stop_token)  # index of the first STOP token
                    decoded_words = decoded_words[: fst_stop_index]
                summaries.append(decoded_words)

                if verbose:
                    decoded_output = " ".join(decoded_words)
                    self.log_output(article, decoded_output)
                

            except Exception as e:
                raise(f"Error on article {i}: {" ".join([word for word in article])}\n\n{e}")
        assert len(examples) == len(summaries), f"Expected number of summaries ({len(summaries)}) to match number of articles ({len(examples)})."

    def log_output(self, article: List[str], summary: str) -> None:
        if self.logger is None:
            raise ValueError(f"Cannot log output without a Logger. Logger: {self.logger}")
        article_text = " ".join(article)
        self.logger.info(f"ARTICLE TEXT: {article_text}\n--------\nSUMMARY: {summary}")
    