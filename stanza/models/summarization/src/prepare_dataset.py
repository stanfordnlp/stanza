"""
Preprocessing on the CNN Dailymail dataset 
"""
import argparse
import os
import logging
import sys
import random

ROOT = '/Users/alexshan/Desktop/stanza'
sys.path.append(ROOT)

from datasets import load_dataset
from enum import Enum
from typing import List, Tuple, Mapping
import stanza
from stanza.server import CoreNLPClient

logger = logging.getLogger('stanza.summarization') 
logger.propagate = False

# Check if the logger has handlers already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


START_TOKEN, STOP_TOKEN = "<s>", "</s>"

class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "validation"


class Chunk:

    def __init__(self, path: str, articles: List[str], summaries: List[str]):

        self.path = path
        if os.path.exists(self.path):
            raise FileExistsError(f"Did not expect chunk path {self.path} to exist.")

        self.articles = articles 
        self.summaries = summaries

        if len(self.articles) != len(self.summaries):
            raise ValueError(f"Expected length of articles ({len(self.articles)}) to match summaries ({len(self.summaries)})")
        
    def extend_chunk(self, article: str, summary: str):
        self.articles = self.articles + [article]
        self.summaries = self.summaries + [summary]
    
    def write_chunk(self, tokenizer_client: CoreNLPClient):
        with open(self.path, "w+", encoding="utf-8") as f:
            for article, summary in zip(self.articles, self.summaries):
                
                processed_article = []
                ann = tokenizer_client.annotate(article)
                for sentence in ann.sentence:
                    for token in sentence.token:
                        processed_article.append(token.word)

                processed_summary = []
                ann = tokenizer_client.annotate(summary)
                for sentence in ann.sentence:
                    for token in sentence.token:
                        processed_summary.append(token.word)

                final_summary = ["<s>"] + processed_summary + ["</s>"]

                f.write(" ".join(processed_article) + "\n")  # article
                f.write(" ".join(final_summary) + "\n")  # summary
        logger.info(f"Succesfully wrote chunk to {self.path}.")

    def __len__(self):
        return len(self.articles)


class Dataset:
    
    def __init__(self, data_root: str, batch_size: int, shuffle: bool = True):
        """
        Args:
            data_root (str): Path to the root of the data directory containing chunked files.
            batch_size (int): Size of each batch of examples 
        
        Returns:
            1. List[List[List[str]]]: Batches of articles, where each token is a separate entry within each sentence
            2. List[List[List[str]]]: Batches of summaries, where each token is a separate entry within each sentence
        """

        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Expected to find path to root: {data_root}, but is missing.")
        
        articles, summaries = [], []
        chunked_files = os.listdir(data_root)
        data_paths = [os.path.join(data_root, chunked) for chunked in chunked_files]

        for path in data_paths:
            with open(path, "r+", encoding='utf-8') as f:
                
                lines = f.readlines()

                for i in range(0, len(lines), 2):  # iterate through lines in increments of two, getting article + summary
                    article, summary = lines[i].strip("\n"), lines[i + 1].strip("\n")
                    tokenized_article, tokenized_summary = article.split(" "), summary.split(" ")
                    articles.append(tokenized_article)
                    summaries.append(tokenized_summary)
        
        self.articles = articles
        self.summaries = summaries

        if len(self.articles) != len(self.summaries):
            raise ValueError(f"Data mismatch: found {len(self.summaries)} summaries compared to {len(self.articles)} articles.")

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        # Number of batches, rounded up to the nearest batch
        return len(self.articles) // self.batch_size + (len(self.articles) % self.batch_size > 0)
    
    def __iter__(self):
        num_articles = len(self.articles)
        indices = list(range(num_articles))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(self.__len__()):
            batch_start = self.batch_size * i 
            batch_end = min(batch_start + self.batch_size, num_articles)

            batch_articles = [self.articles[x] for x in indices[batch_start: batch_end]]
            batch_summaries = [self.summaries[x] for x in indices[batch_start: batch_end]]
            yield batch_articles, batch_summaries
        

def write_dataset(save_path_root: str, split: DatasetSplit, streaming: bool = True, chunk_size: int = 1000):
    """
    Writes the `split` section of the CNN dailymail dataset to `save_path_root`. The dataset will be split into
    chunked files, where the path to the data will be `{save_path_root}/{split}/examples_{i}` for the i-th chunk.

    Args:
        save_path (str): Path to save file

    """
    
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split.value, streaming=streaming)

    chunk_number = 0
    save_name = os.path.join(save_path_root, split.value,f"examples_{chunk_number}.txt")
    cur_chunk = Chunk(save_name, [], [])  # empty chunk
    with CoreNLPClient(annotators=['tokenize'], timeout=60000, be_quiet=True) as client:
        try:  # attempt to iterate through dataset
            for example in dataset:
                article, summary = example.get("article"), example.get("highlights")
                cur_chunk.extend_chunk(article, summary)
                
                if len(cur_chunk) == chunk_size:  # chunk finished, write out results
                    cur_chunk.write_chunk(tokenizer_client=client)
                    chunk_number += 1
                    save_name = os.path.join(save_path_root, split.value,f"examples_{chunk_number}.txt")
                    cur_chunk = Chunk(save_name, [], [])  # empty chunk
                    
        except StopIteration:
            # Write any remaining examples to final chunk
            cur_chunk.write_chunk(tokenizer_client=client)
            logger.info("All examples have been processed.")

def split_type(value):
    try:
        return DatasetSplit(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid DatasetSplit type. Choose from 'train', 'validation', 'test'.")


if __name__ == "__main__":

    ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    SPLIT = DatasetSplit.TRAIN

    write_dataset(ROOT, SPLIT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=ROOT, help="Path to root directory containing data for the split you are preparing dataset for.")
    parser.add_argument("--split", type=split_type, required=True, default=SPLIT, help="Which split you are attempting to write data for (train, validation, test)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of data chunk files to split data into.")

    args = parser.parse_args()

    data_root = args.data_root
    split = args.split
    chunk_size = args.chunk_size

    logger.info(f"Using the following args for preparing dataset: ")
    for k, v in args.items():
        logger.info(f"{k}: {v}")

    write_dataset(data_root, split, chunk_size=chunk_size)