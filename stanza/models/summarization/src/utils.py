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


def convert_text_to_token_ids(vocab_map: Mapping[str, int], text_batch: List[List[str]], UNK_ID: int, max_steps: int = None):
    """
    Converts a text batch to a batch of token IDs.
    """

    token_ids = []

    if max_steps is not None:  # Truncate
         text_batch = [article[: max_steps] for article in text_batch]

    for article in text_batch:
        article_token_ids = torch.tensor([vocab_map.get(word.lower(), UNK_ID) for word in article])
        token_ids.append(article_token_ids)

    padded_inputs = pad_sequence(token_ids, batch_first=True)
    return padded_inputs

def generate_checkpoint_path(save_path: str):

    """
    Given a model path, create a checkpoint path by appending '_ckpt' to the filename.

    Args:
    model_path (str): The original path to the model file.

    Returns:
    str: The new path with '_ckpt' appended to the filename.
    """
    # Extract the directory and filename
    dir_name = os.path.dirname(save_path)
    base_name = os.path.basename(save_path)
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(base_name)
    
    # Create the new filename with '_ckpt' appended
    new_base_name = f"{name}_ckpt{ext}"
    
    # Construct the new path
    new_path = os.path.join(dir_name, new_base_name)
    
    return new_path