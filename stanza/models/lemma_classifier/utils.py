import stanza
import torch
import os
import prepare_dataset

from stanza.models.lemma_classifier.constants import DEFAULT_BATCH_SIZE
from typing import List, Tuple, Any, Mapping
from collections import Counter, defaultdict    


def load_dataset(data_path: str, batch_size=DEFAULT_BATCH_SIZE, get_counts: bool = False, label_decoder: dict = None) -> Tuple[List[List[str]], List[torch.Tensor], List[torch.Tensor], Mapping[int, int], Mapping[str, int]]:
    """
    Loads a data file into data batches for tokenized text sentences, token indices, and true labels for each sentence.

    Args:
        data_path (str): Path to data file, containing tokenized text sentences, token index and true label for token lemma on each line. 
        batch_size (int): Size of each batch of examples
        get_counts (optional, bool): Whether there should be a map of the label index to counts

    Returns:
        1. List[List[List[str]]]: Batches of sentences, where each token is a separate entry in each sentence
        2. List[torch.tensor[int]]: A batch of indexes for the target token corresponding to its sentence
        3. List[torch.tensor[int]]: A batch of labels for the target token's lemma
        4 (Optional): A mapping of label ID to counts in the dataset.
        5. Mapping[str, int]: A map between the labels and their indexes

    """

    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} could not be found.")

    if label_decoder is None:
        label_decoder = {}
    else:
        # if labels in the test set aren't in the original model,
        # the model will never predict those labels,
        # but we can still use those labels in a confusion matrix
        label_decoder = dict(label_decoder)

    with open(data_path, "r+", encoding="utf-8") as f:
        sentences, indices, labels, counts = [], [], [], Counter()
        for line in f.readlines():
            line_contents = line.split()
            if not line_contents:
                continue
            
            sentence = line_contents[: -2]
            index, label = line_contents[-2:]

            index = int(index)

            label_id = label_decoder.get(label, None)
            if label_id is None:
                label_decoder[label] = len(label_decoder)

            sentences.append(sentence)
            indices.append(index)
            labels.append(label_decoder[label])

            if get_counts:
                counts[label_decoder[label]] += 1

    sentence_batches = [sentences[i: i + batch_size] for i in range(0, len(sentences), batch_size)]
    indices_batches = [torch.tensor(indices[i: i + batch_size]) for i in range(0, len(indices), batch_size)]
    labels_batches = [torch.tensor(labels[i: i + batch_size]) for i in range(0, len(indices), batch_size)]
    
    return sentence_batches, indices_batches, labels_batches, counts, label_decoder


def extract_unknown_token_indices(tokenized_indices: torch.tensor, unknown_token_idx: int) -> List[int]:
    """
    Extracts the indices within `tokenized_indices` which match `unknown_token_idx`

    Args:
        tokenized_indices (torch.tensor): A tensor filled with tokenized indices of words that have been mapped to vector indices.
        unknown_token_idx (int): The special index for which unknown tokens are marked in the word vectors.

    Returns:
        List[int]: A list of indices in `tokenized_indices` which match `unknown_token_index`
    """
    return [idx for idx, token_index in enumerate(tokenized_indices) if token_index == unknown_token_idx]


def get_device():
    """
    Get the device to run computations on
    """
    if torch.cuda.is_available:
        device = torch.device("cuda")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device
