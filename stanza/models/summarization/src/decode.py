"""
Takes an existing model and runs beam search decoding across many examples

"""

import torch

from copy import deepcopy
from typing import List, Tuple, Mapping, Any
from stanza.models.summarization.src.model import BaselineSeq2Seq
from stanza.models.summarization.src.beam_search import *
from stanza.models.common.vocab import BaseVocab, UNK
from logging import Logger
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

logger = logging.getLogger('stanza.summarization') 
logger.propagate = False

# Check if the logger has handlers already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class BeamSearchDecoder():

    """
    Decoder for summarization using beam search
    """

    def __init__(self, model: BaselineSeq2Seq):
        self.model = model 
        self.stop_token = "</s>"
        self.start_token = "<s>" 
        self.ext_id2unit = {idx: word for word, idx in self.model.ext_vocab_map.items()}
        self.ext_unit2id = self.model.ext_vocab_map

        logger.info(f"Loaded model into BeamSearchDecoder on device {next(self.model.parameters()).device}")
    
    def decode_examples(self, examples: List[List[str]], beam_size: int, max_dec_steps: int = None, min_dec_steps: int = None,
                        max_enc_steps: int = None, verbose: bool = True) -> List[List[str]]:
        summaries = []  # outputs 

        for i, article in tqdm(enumerate(examples), desc="decoding examples for evaluation..."):
            
            try:
                # Run beam search to get the best hypothesis
                best_hyp, id2unit = run_beam_search(self.model, 
                                           self.ext_unit2id,
                                           self.ext_id2unit, 
                                           article, 
                                           beam_size=beam_size,
                                           max_dec_steps=max_dec_steps,
                                           min_dec_steps=min_dec_steps,
                                           max_enc_steps=max_enc_steps,
                                           )
                
                output_ids = [int(t) for t in best_hyp.tokens[1: ]]  # exclude START tokens but not STOP because not guaranteed to contain STOP

                decoded_words = [id2unit.get(idx) for idx in output_ids]
                if self.stop_token in decoded_words:
                    fst_stop_index = decoded_words.index(self.stop_token)  # index of the first STOP token
                    decoded_words = decoded_words[: fst_stop_index]
                summaries.append(decoded_words)

                if verbose:
                    decoded_output = " ".join(decoded_words)
                    self.log_output(article, decoded_output)
                

            except Exception as e:
                logger.error(f'Error on article {i}: {" ".join([word for word in article])}\n')
                raise(e)
        assert len(examples) == len(summaries), f"Expected number of summaries ({len(summaries)}) to match number of articles ({len(examples)})."
        return summaries
    
    def log_output(self, article: List[str], summary: str) -> None:
        if logger is None:
            raise ValueError(f"Cannot log output without a Logger. Logger: {logger}")
        article_text = " ".join(article)
        logger.info(f"ARTICLE TEXT: {article_text}\n--------\nSUMMARY: {summary}")
    