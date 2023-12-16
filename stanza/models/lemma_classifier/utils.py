import stanza
import torch
import os
from typing import List, Tuple, Any, Mapping

def load_doc_from_conll_file(path: str):
    """"
    loads in a Stanza document object from a path to a CoNLL file containing annotated sentences.
    """
    return stanza.utils.conll.CoNLL.conll2doc(path)


def load_dataset(data_path: str, label_decoder: Mapping[str, int], get_counts: bool = False) -> Tuple[List[List[str]], List[int], List[int], Mapping[int, int]]:

    """
    Loads a data file into data batches for tokenized text sentences, token indices, and true labels for each sentence.

    Args:
        data_path (str): Path to data file, containing tokenized text sentences, token index and true label for token lemma on each line. 
        label_decoder (Mapping[str, int]): A map between target token lemmas and their corresponding integers for the labels
        get_counts (optional, bool): Whether there should be a map of the label index to counts

    Returns:
        1. List[List[str]]: A list of sentences, where each token is a separate entry
        2. List[int]: A list of indexes for the target token corresponding to its sentence
        3. List[int]: A list of labels for the target token's lemma
        4 (Optional): A mapping of label ID to counts in the dataset.

    """

    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} could not be found.")
    if not label_decoder:
        raise ValueError(f"Label decoder {label_decoder} is invalid.")

    sentences, indices, labels, counts = [], [], [], {}

    with open(data_path, "r+", encoding="utf-8") as f:
        for line in f.readlines():
            line_contents = line.split()
            if not line_contents:
                continue
            
            sentence = line_contents[: -2]
            index, label = line_contents[-2:]

            index = int(index)

            label_id = label_decoder.get(label, None)
            if label_id is None:
                ValueError(f"Label {label} was not found in the label decoder ({label_decoder}).")

            sentences.append(sentence)
            indices.append(index)
            labels.append(label_id)

            if get_counts:
                if label_id not in counts:
                    counts[label_id] = 0
                counts[label_id] += 1 
    
    return sentences, indices, labels, counts


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


