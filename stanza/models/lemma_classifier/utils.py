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
        4. List[torch.tensor[int]]: A batch of UPOS IDs for the target token
        5 (Optional): A mapping of label ID to counts in the dataset.
        6. Mapping[str, int]: A map between the labels and their indexes
        7. Mapping[str, int]: A map between the UPOS tags and their corresponding IDs found in the UPOS batches
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
        sentences, indices, labels, upos_ids, counts, upos_to_id = [], [], [], [], Counter(), defaultdict(str)
        data_processor = prepare_dataset.DataProcessor("", [], "")
        sentences_data = data_processor.read_processed_data(data_path)

        for idx, sentence in enumerate(sentences_data):
            # TODO Could replace this with sentence.values(), but need to know if Stanza requires Python 3.7 or later for backward compatability reasons
            words, target_idx, target_upos, label = sentence.get("words"), sentence.get("index"), sentence.get("upos"), sentence.get("lemma")   
            if None in [words, target_idx, target_upos, label]:
                raise ValueError(f"Expected data to be complete but found a null value in sentence {idx}: {sentence}")
            
            label_id = label_decoder.get(label, None)
            if label_id is None:
                label_decoder[label] = len(label_decoder)  # create a new ID for the unknown label

            upos_id = upos_to_id.get(target_upos, None)
            if upos_id is None:
                upos_to_id[target_upos] = len(upos_to_id)  # create a new ID for the unknown upos tag

            sentences.append(words)
            indices.append(target_idx)
            upos_ids.append(upos_to_id[target_upos])
            labels.append(label_decoder[label])

            if get_counts:
                counts[label_decoder[label]] += 1

    sentence_batches = [sentences[i: i + batch_size] for i in range(0, len(sentences), batch_size)]
    indices_batches = [torch.tensor(indices[i: i + batch_size]) for i in range(0, len(indices), batch_size)]
    upos_batches = [torch.tensor(upos_ids[i: i + batch_size]) for i in range(0, len(upos_ids), batch_size)]
    labels_batches = [torch.tensor(labels[i: i + batch_size]) for i in range(0, len(labels), batch_size)]
    # TODO consider making the return object a JSON or a custom object for cleaner access instead of a big tuple of stuff
    return sentence_batches, indices_batches, upos_batches, labels_batches, counts, label_decoder, upos_to_id


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


def main():
    default_test_path = os.path.join(os.path.dirname(__file__), "test_sets", "with_upos_processed_ewt_dev.txt")   # get the GUM stuff
    sentence_batches, indices_batches, upos_batches, _, counts, _, _ = load_dataset(default_test_path, get_counts=True)
    print(upos_batches)
    print(counts)

if __name__ == "__main__":
    main()