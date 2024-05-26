"""
Preprocessing on the CNN Dailymail dataset 
"""
import os
import logging
import sys

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


class Chunk():

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
        with open(self.path, "w+") as f:
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


def write_dataset(save_path_root: str, split: DatasetSplit, streaming: bool = True, chunk_size: int = 1000):
    """
    Writes the `split` section of the CNN dailymail dataset to `save_path_root`. The dataset will be split into
    chunked files, where the path to the data will be {save_path_root}/{split}/examples_{i} for the i-th chunk.

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


if __name__ == "__main__":

    ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    SPLIT = DatasetSplit.TRAIN

    write_dataset(ROOT, SPLIT)
