from collections import Counter
import json
import logging
import os
import random
from typing import List, Tuple, Any, Mapping

import stanza
import torch

from stanza.models.lemma_classifier.constants import DEFAULT_BATCH_SIZE

logger = logging.getLogger('stanza.lemmaclassifier')

class Dataset:
    def __init__(self, data_path: str, batch_size: int =DEFAULT_BATCH_SIZE, get_counts: bool = False, label_decoder: dict = None, shuffle: bool = True):
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
            4. List[List[int]]: A batch of UPOS IDs for the target token (this is a List of Lists, not a tensor. It should be padded later.)
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

        logger.debug("Final label decoder: %s  Should be strings to ints", label_decoder)

        # words which we are analyzing
        target_words = set()

        # all known words in the dataset, not just target words
        known_words = set()

        with open(data_path, "r+", encoding="utf-8") as fin:
            sentences, indices, labels, upos_ids, counts, upos_to_id = [], [], [], [], Counter(), {}

            input_json = json.load(fin)
            sentences_data = input_json['sentences']
            self.target_upos = input_json['upos']

            for idx, sentence in enumerate(sentences_data):
                # TODO Could replace this with sentence.values(), but need to know if Stanza requires Python 3.7 or later for backward compatability reasons
                words, target_idx, upos_tags, label = sentence.get("words"), sentence.get("index"), sentence.get("upos_tags"), sentence.get("lemma")
                if None in [words, target_idx, upos_tags, label]:
                    raise ValueError(f"Expected data to be complete but found a null value in sentence {idx}: {sentence}")

                label_id = label_decoder.get(label, None)
                if label_id is None:
                    label_decoder[label] = len(label_decoder)  # create a new ID for the unknown label

                converted_upos_tags = []  # convert upos tags to upos IDs
                for upos_tag in upos_tags:
                    if upos_tag not in upos_to_id:
                        upos_to_id[upos_tag] = len(upos_to_id)  # create a new ID for the unknown UPOS tag
                    converted_upos_tags.append(upos_to_id[upos_tag])

                sentences.append(words)
                indices.append(target_idx)
                upos_ids.append(converted_upos_tags)
                labels.append(label_decoder[label])

                if get_counts:
                    counts[label_decoder[label]] += 1

                target_words.add(words[target_idx])
                known_words.update(words)

        self.sentences = sentences
        self.indices = indices
        self.upos_ids = upos_ids
        self.labels = labels

        self.counts = counts
        self.label_decoder = label_decoder
        self.upos_to_id = upos_to_id

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.known_words = [x.lower() for x in sorted(known_words)]
        self.target_words = set(x.lower() for x in target_words)

    def __len__(self):
        """
        Number of batches, rounded up to nearest batch
        """
        return len(self.sentences) // self.batch_size + (len(self.sentences) % self.batch_size > 0)

    def __iter__(self):
        num_sentences = len(self.sentences)
        indices = list(range(num_sentences))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(self.__len__()):
            batch_start = self.batch_size * i
            batch_end = min(batch_start + self.batch_size, num_sentences)

            batch_sentences = [self.sentences[x] for x in indices[batch_start:batch_end]]
            batch_indices =   torch.tensor([self.indices[x] for x in indices[batch_start:batch_end]])
            batch_upos_ids =  [self.upos_ids[x] for x in indices[batch_start:batch_end]]
            batch_labels =    torch.tensor([self.labels[x] for x in indices[batch_start:batch_end]])
            yield batch_sentences, batch_indices, batch_upos_ids, batch_labels

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


def round_up_to_multiple(number, multiple):
    if multiple == 0:
        return "Error: The second number (multiple) cannot be zero."

    # Calculate the remainder when dividing the number by the multiple
    remainder = number % multiple

    # If remainder is non-zero, round up to the next multiple
    if remainder != 0:
        rounded_number = number + (multiple - remainder)
    else:
        rounded_number = number  # No rounding needed

    return rounded_number


def main():
    default_test_path = os.path.join(os.path.dirname(__file__), "test_sets", "processed_ud_en", "combined_dev.txt")   # get the GUM stuff
    sentence_batches, indices_batches, upos_batches, _, counts, _, upos_to_id = load_dataset(default_test_path, get_counts=True)

if __name__ == "__main__":
    main()
