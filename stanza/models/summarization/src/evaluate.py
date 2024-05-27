"""
Evaluates a trained abstractive summarization Seq2Seq model
"""
import sys
import os 
import torch

ROOT = '/Users/alexshan/Desktop/stanza'
sys.path.append(ROOT)

import evaluate
import logging
from typing import List, Tuple, Mapping
from stanza.models.summarization.src.decode import BeamSearchDecoder
from stanza.models.summarization.src.model import BaselineSeq2Seq
from stanza.models.common.vocab import BaseVocab


logger = logging.getLogger('stanza.summarization') 
logger.propagate = False

# Check if the logger has handlers already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def evaluate_predictions_rouge(generated_summaries: List[str], reference_summaries: List[str]):
    
    rouge = evaluate.load('rouge')
    results = rouge.compute(
                            predictions=generated_summaries,
                            references=reference_summaries
                            )
    
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")


def evaluate_model(model_path: str, articles: List[List[str]], summaries: List[List[str]], logger: logging.Logger = None):
    
    trained_model = torch.load(model_path)
    vocab = trained_model.vocab

    decoder = BeamSearchDecoder(trained_model, vocab, logger)

    generated_summaries = decoder.decode_examples(
                                                 examples=articles,
                                                 beam_size=4,
                                                 max_dec_steps=400,
                                                 min_dec_steps=10,
                                                 verbose=False       
                                                 )  # TODO consider making these toggle-able via argparse
    
    generated_summaries = [" ".join(summary) for summary in generated_summaries]
    summaries = [" ".join(summary) for summary in summaries]
    
    evaluate_predictions_rouge(generated_summaries, summaries)


def evaluate_from_path(model_path: str, eval_path: str, logger: logging.Logger = None):
    
    # Get data
    articles, summaries = [], []
    chunked_files = os.listdir(eval_path)
    data_paths = [os.path.join(eval_path, chunked) for chunked in chunked_files]

    for path in data_paths:  # TODO consider moving this section as a helper
        with open(path, "r+", encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):  # iterate through lines in increments of two, getting article + summary
                article, summary = lines[i].strip("\n"), lines[i + 1].strip("\n")
                tokenized_article, tokenized_summary = article.split(" "), summary.split(" ")
                articles.append(tokenized_article)
                summaries.append(tokenized_summary)
    
    evaluate_model(
                   model_path, 
                   articles, 
                   summaries, 
                   logger
                   )


# TODO: add argparse