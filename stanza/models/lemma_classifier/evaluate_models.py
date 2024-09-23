import os
import sys

parentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import logging
import argparse
import os

from typing import Any, List, Tuple, Mapping
from collections import defaultdict
from numpy import random

import torch
import torch.nn as nn

import stanza

from stanza.models.common.utils import default_device
from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.lstm_model import LemmaClassifierLSTM
from stanza.models.lemma_classifier.transformer_model import LemmaClassifierWithTransformer
from stanza.utils.confusion import format_confusion
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

logger = logging.getLogger('stanza.lemmaclassifier')


def get_weighted_f1(mcc_results: Mapping[int, Mapping[str, float]], confusion: Mapping[int, Mapping[int, int]]) -> float:
    """
    Computes the weighted F1 score across an evaluation set.

    The weight of a class's F1 score is equal to the number of examples in evaluation. This makes classes that have more
    examples in the evaluation more impactful to the weighted f1.
    """
    num_total_examples = 0
    weighted_f1 = 0

    for class_id in mcc_results:
        class_f1 = mcc_results.get(class_id).get("f1")
        num_class_examples = sum(confusion.get(class_id).values())
        weighted_f1 += class_f1 * num_class_examples
        num_total_examples += num_class_examples

    return weighted_f1 / num_total_examples


def evaluate_sequences(gold_tag_sequences: List[Any], pred_tag_sequences: List[Any], label_decoder: Mapping, verbose=True):
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

    reverse_label_decoder = {y: x for x, y in label_decoder.items()}
    for gold, pred in zip(gold_tag_sequences, pred_tag_sequences):
        confusion[reverse_label_decoder[gold]][reverse_label_decoder[pred]] += 1

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
            logger.info(f"Lemma '{lemma}' had precision {100 * multi_class_result[lemma]['precision']}, recall {100 * multi_class_result[lemma]['recall']} and F1 score of {100 * multi_class_result[lemma]['f1']}")

    weighted_f1 = get_weighted_f1(multi_class_result, confusion)

    return multi_class_result, confusion, weighted_f1


def model_predict(model: nn.Module, position_indices: torch.Tensor, sentences: List[List[str]], upos_tags: List[List[int]]=[]) -> torch.Tensor:
    """
    A LemmaClassifierLSTM or LemmaClassifierWithTransformer is used to predict on a single text example, given the position index of the target token.

    Args:
        model (LemmaClassifier): A trained LemmaClassifier that is able to predict on a target token.
        position_indices (Tensor[int]): A tensor of the (zero-indexed) position of the target token in `text` for each example in the batch.
        sentences (List[List[str]]): A list of lists of the tokenized strings of the input sentences.

    Returns:
        (int): The index of the predicted class in `model`'s output.
    """
    with torch.no_grad():
        logits = model(position_indices, sentences, upos_tags)  # should be size (batch_size, output_size)
        predicted_class = torch.argmax(logits, dim=1)  # should be size (batch_size, 1)

    return predicted_class


def evaluate_model(model: nn.Module, eval_path: str, verbose: bool = True, is_training: bool = False) -> Tuple[Mapping, Mapping, float, float]:
    """
    Helper function for model evaluation

    Args:
        model (LemmaClassifierLSTM or LemmaClassifierWithTransformer): An instance of the LemmaClassifier class that has architecture initialized which matches the model saved in `model_path`.
        model_path (str): Path to the saved model weights that will be loaded into `model`.
        eval_path (str): Path to the saved evaluation dataset.
        verbose (bool, optional): True if `evaluate_sequences()` should print the F1, Precision, and Recall for each class. Defaults to True.
        is_training (bool, optional): Whether the model is in training mode. If the model is training, we do not change it to eval mode.

    Returns:
        1. Multi-class results (Mapping[int, Mapping[str, float]]): first map has keys as the classes (lemma indices) and value is
                                                                    another map with key of "f1", "precision", or "recall" with corresponding values.
        2. Confusion Matrix (Mapping[int, Mapping[int, int]]): A confusion matrix with keys equal to the index of the gold tag, and a value of the
                                                               map with the key as the predicted tag and corresponding count of that (gold, pred) pair.
        3. Accuracy (float): the total accuracy (num correct / total examples) across the evaluation set.
    """
    # load model
    device = default_device()
    model.to(device)

    if not is_training:
        model.eval()  # set to eval mode

    # load in eval data
    dataset = utils.Dataset(eval_path, label_decoder=model.label_decoder, shuffle=False)

    logger.info(f"Evaluating on evaluation file {eval_path}")

    correct, total = 0, 0
    gold_tags, pred_tags = dataset.labels, []

    # run eval on each example from dataset
    for sentences, pos_indices, upos_tags, labels in tqdm(dataset, "Evaluating examples from data file"):
        pred = model_predict(model, pos_indices, sentences, upos_tags)  # Pred should be size (batch_size, )
        correct_preds = pred == labels.to(device)
        correct += torch.sum(correct_preds)
        total += len(correct_preds)
        pred_tags += pred.tolist()

    logger.info("Finished evaluating on dataset. Computing scores...")
    accuracy = correct / total

    mc_results, confusion, weighted_f1 = evaluate_sequences(gold_tags, pred_tags, dataset.label_decoder, verbose=verbose)
    # add brackets around batches of gold and pred tags because each batch is an element within the sequences in this helper
    if verbose:
        logger.info(f"Accuracy: {accuracy} ({correct}/{total})")
        logger.info(f"Label decoder: {dataset.label_decoder}")

    return mc_results, confusion, accuracy, weighted_f1


def main(args=None, predefined_args=None):

    # TODO: can unify this script with train_lstm_model.py?
    # TODO: can save the model type in the model .pt, then
    # automatically figure out what type of model we are using by
    # looking in the file
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000, help="Number of tokens in vocab")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Number of dimensions in word embeddings (currently using GloVe)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Size of hidden layer")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument("--charlm", action='store_true', default=False, help="Whether not to use the charlm embeddings")
    parser.add_argument('--charlm_shorthand', type=str, default=None, help="Shorthand for character-level language model training corpus.")
    parser.add_argument("--charlm_forward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_forward.pt"), help="Path to forward charlm file")
    parser.add_argument("--charlm_backward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_backwards.pt"), help="Path to backward charlm file")
    parser.add_argument("--save_name", type=str, default=os.path.join(os.path.dirname(__file__), "saved_models", "lemma_classifier_model.pt"), help="Path to model save file")
    parser.add_argument("--model_type", type=str, default="roberta", help="Which transformer to use ('bert' or 'roberta' or 'lstm')")
    parser.add_argument("--bert_model", type=str, default=None, help="Use a specific transformer instead of the default bert/roberta")
    parser.add_argument("--eval_file", type=str, help="path to evaluation file")

    args = parser.parse_args(args) if not predefined_args else predefined_args

    logger.info("Running training script with the following args:")
    args = vars(args)
    for arg in args:
        logger.info(f"{arg}: {args[arg]}")
    logger.info("------------------------------------------------------------")

    logger.info(f"Attempting evaluation of model from {args['save_name']} on file {args['eval_file']}")
    model = LemmaClassifier.load(args['save_name'], args)

    mcc_results, confusion, acc, weighted_f1 = evaluate_model(model, args['eval_file'])

    logger.info(f"MCC Results: {dict(mcc_results)}")
    logger.info("______________________________________________")
    logger.info(f"Confusion:\n%s", format_confusion(confusion))
    logger.info("______________________________________________")
    logger.info(f"Accuracy: {acc}")
    logger.info("______________________________________________")
    logger.info(f"Weighted f1: {weighted_f1}")

    return mcc_results, confusion, acc, weighted_f1


if __name__ == "__main__":
    main()
