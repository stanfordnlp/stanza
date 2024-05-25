"""
Utility functions for building and training summarization model(s)
"""
import torch
from itertools import islice
from typing import List, Tuple, Mapping
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def generate_train_subset():
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
            # Tokenize both the article and the highlights to get raw tokens
            tokens_article = [tokenizer.tokenize(article) for article in examples["article"]]
            tokens_highlights = [tokenizer.tokenize(highlights) for highlights in examples["highlights"]]
            
            # Combine the tokenized fields into a single dictionary
            return {
                "tokens_article": tokens_article,
                "tokens_highlights": tokens_highlights,
            }
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=16)

    for example in tokenized_dataset:
        input_text = example['article']
        summary_text = example['highlights']


def convert_text_to_token_ids(vocab_map: Mapping[str, int], text_batch: List[List[str]], UNK_ID: int):
    """
    Converts a text batch to a batch of token IDs.
    """

    token_ids = []
    for article in text_batch:
        article_token_ids = torch.tensor([vocab_map.get(word.lower(), UNK_ID) for word in article])
        token_ids.append(article_token_ids)

    padded_inputs = pad_sequence(token_ids, batch_first=True)
    return padded_inputs