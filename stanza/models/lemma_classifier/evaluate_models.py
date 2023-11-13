import os 
import sys 

parentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import stanza
import torch
import utils

from typing import Any, List, Tuple, Mapping
from collections import defaultdict
from constants import get_glove
from model import LemmaClassifier
from constants import *
from tqdm import tqdm
from numpy import random


def evaluate_sequences(gold_tag_sequences: List[List[Any]], pred_tag_sequences: List[List[Any]], verbose=True):
    """
    Evaluates a model's predicted tags against a set of gold tags. Computes precision, recall, and f1 for all classes.

    Precision = true positives / true positives + false positives
    Recall = true positives / true positives + false negatives
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Returns:
        1. Multi class result dictionary, where each class is a key and maps to another map of its F1, precision, and recall scores.
           e.g. multiclass_results[0]["precision"] would give class 0's precision.
        2. Confusion matrix, where each key is a gold tag and its value is another map with a key of the predicted tag with value of that (gold, pred) count.
           e.g. confusion[0][1] = 6 would mean that for gold tag 0, the model predicted tag 1 a total of 6 times.
    """
    assert len(gold_tag_sequences) == len(pred_tag_sequences), \
    f"Length of gold tag sequences is {len(gold_tag_sequences)}, while length of predicted tag sequence is {len(pred_tag_sequences)}"        
    
    confusion = defaultdict(lambda: defaultdict(int))
    
    for gold_tags, pred_tags in tqdm(zip(gold_tag_sequences, pred_tag_sequences), "Evaluating sequences"):

        assert len(gold_tags) == len(pred_tags), f"Number of gold tags doesn't match number of predicted tags ({len(gold_tags)}, {len(pred_tags)})"
        for gold, pred in zip(gold_tags, pred_tags):
            confusion[gold][pred] += 1

    multi_class_result = defaultdict(lambda: defaultdict(float))
    # compute precision, recall and f1 for each class and store inside of `multi_class_result`
    for gold_tag in confusion.keys():

        try:
            prec = confusion.get(gold_tag, {}).get(gold_tag, 0) / sum([confusion.get(k, {}).get(gold_tag, 0) for k in confusion.keys()])
        except ZeroDivisionError:
            prec = 0.0 
        
        try:
            recall = confusion.get(gold_tag, {}).get(gold_tag, 0) / sum(confusion.get(gold_tag, {}).values())
        except ZeroDivisionError:
            recall = 0.0

        try:
            f1 = 2 * (prec * recall) / (prec + recall)
        except ZeroDivisionError:
            f1 = 0.0 

        multi_class_result[gold_tag] = {
            "precision": prec,
            "recall": recall,
            "f1": f1
        }
    
    if verbose:
        for lemma in multi_class_result:
            print(f"Lemma '{lemma}' had precision {100 * multi_class_result[lemma]['precision']}, recall {100 * multi_class_result[lemma]['recall']} and F1 score of {100 * multi_class_result[lemma]['f1']}")

    return multi_class_result, confusion   


def model_predict(model: LemmaClassifier, text: List[int], position_idx: int) -> int:
    """
    A LemmaClassifier is used to predict on a single text example, given the position index of the target token.

    Args:
        model (LemmaClassifier): A trained LemmaClassifier that is able to predict on a target token.
        text (List[int]): A tokenized sentence with the proper embeddings corresponding to `model`.
        position_idx (int): The (zero-indexed) position of the target token in `text`.
    
    Returns:
        (int): The index of the predicted class in `model`'s output.
    """
    assert len(text) != 0, f"Text arg is empty. Please provide a proper input for model evaluation."
    if not isinstance(text[0], int):
        raise TypeError(f"Text variable must contain tokenized version of sentence, but instead found type {type(text[0])}.")


    text_tensor = torch.tensor(text)
    with torch.no_grad():
        logits = model(text_tensor, position_idx)
        predicted_class = torch.argmax(logits).item()
    
    return predicted_class


def evaluate_model(model: LemmaClassifier, model_path: str, eval_path: str, label_decoder: Mapping[str, int], 
                   verbose: bool = True) -> Tuple[Mapping, Mapping, float]:
    """
    Helper function for model evaluation

    Args:
        model (LemmaClassifier): An instance of the LemmaClassifier class that has architecture initialized which matches the model saved in `model_path`.
        model_path (str): Path to the saved model weights that will be loaded into `model`.
        eval_path (str): Path to the saved evaluation dataset.
        label_decoder (Mapping[str, int]): A map between target token lemmas and their corresponding integers for the labels
        verbose (bool, optional): True if `evaluate_sequences()` should print the F1, Precision, and Recall for each class. Defaults to True.

    Returns:
        1. Multi-class results (Mapping[int, Mapping[str, float]]): first map has keys as the classes (lemma indices) and value is 
                                                                    another map with key of "f1", "precision", or "recall" with corresponding values.
        2. Confusion Matrix (Mapping[int, Mapping[int, int]]): A confusion matrix with keys equal to the index of the gold tag, and a value of the 
                                                               map with the key as the predicted tag and corresponding count of that (gold, pred) pair.
        3. Accuracy (float): the total accuracy (num correct / total examples) across the evaluation set.
    """
    # load model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # set to eval mode

    # load in eval data 
    text_batches, index_batches, label_batches = utils.load_dataset(eval_path, label_decoder=label_decoder)
    
    correct = 0
    gold_tags, pred_tags = [label_batches], []
    # run eval on each example from dataset
    for sentence, pos_index, label in tqdm(zip(text_batches, index_batches, label_batches), "Evaluating examples from data file"):
        # tokenize raw text sentence using model
        GLOVE = get_glove(model.embedding_dim)   # TODO make this dynamic

        # TODO: See if John approves of this fix
        tokenized_sentence = [GLOVE.stoi[word.lower()] if word.lower() in GLOVE.stoi else UNKNOWN_TOKEN_IDX for word in sentence]  # handle unknown tokens by randomizing their embedding

        pred = model_predict(model, tokenized_sentence, pos_index)
        correct += 1 if pred == label else 0 
        pred_tags += [pred]

    print("Finished evaluating on dataset. Computing scores...")
    accuracy = correct / len(label_batches)
    mc_results, confusion = evaluate_sequences(gold_tags, [pred_tags], verbose=verbose)  
    # add brackets around batches of gold and pred tags because each batch is an element within the sequences in this helper
    if verbose:
        print(f"Accuracy: {accuracy} ({correct}/{len(label_batches)})")
    
    return mc_results, confusion, accuracy



def main():

    vocab_size = 10000  # Adjust based on your dataset
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 2  # Binary classification (be or have)

    model = LemmaClassifier(vocab_size=vocab_size,
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            embeddings=get_glove(embedding_dim).vectors)
    
    model_path = os.path.join(os.path.dirname(__file__), "big_demo_model.pt")
    eval_path = os.path.join(os.path.dirname(__file__), "test_output.txt")
    label_decoder = {"be": 0, "have": 1}

    mcc_results, confusion, acc = evaluate_model(model, model_path, eval_path, label_decoder)

    print(f"MCC Results: {dict(mcc_results)}")
    print("______________________________________________")
    print(f"Confusion: {dict(confusion)}")
    print("______________________________________________")
    print(f"Accuracy: {acc}")

    

if __name__ == "__main__":
    main()
