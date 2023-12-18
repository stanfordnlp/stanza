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

from tqdm import tqdm
import torch

import stanza

from stanza.models.lemma_classifier.constants import get_glove
from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.constants import *
from stanza.models.lemma_classifier.model import LemmaClassifier
from stanza.models.lemma_classifier.transformer_baseline.model import LemmaClassifierWithTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            logging.info(f"Lemma '{lemma}' had precision {100 * multi_class_result[lemma]['precision']}, recall {100 * multi_class_result[lemma]['recall']} and F1 score of {100 * multi_class_result[lemma]['f1']}")

    return multi_class_result, confusion   


def model_predict(model: LemmaClassifier, text: List[int], position_idx: int, words: List[str]) -> int:
    """
    A LemmaClassifier is used to predict on a single text example, given the position index of the target token.

    Args:
        model (LemmaClassifier): A trained LemmaClassifier that is able to predict on a target token.
        text (List[int]): A tokenized sentence with the proper embeddings corresponding to `model`.
        position_idx (int): The (zero-indexed) position of the target token in `text`.
        words (List[str]): A list of the tokenized strings of the input sentence.
    
    Returns:
        (int): The index of the predicted class in `model`'s output.
    """
    assert len(text) != 0, f"Text arg is empty. Please provide a proper input for model evaluation."
    if not isinstance(text[0], int):
        raise TypeError(f"Text variable must contain tokenized version of sentence, but instead found type {type(text[0])}.")


    text_tensor = torch.tensor(text)
    with torch.no_grad():
        logits = model(text_tensor, position_idx, words)
        predicted_class = torch.argmax(logits).item()
    
    return predicted_class


def evaluate_model(model: LemmaClassifier, model_path: str, eval_path: str, verbose: bool = True) -> Tuple[Mapping, Mapping, float]:
    """
    Helper function for model evaluation

    Args:
        model (LemmaClassifier): An instance of the LemmaClassifier class that has architecture initialized which matches the model saved in `model_path`.
        model_path (str): Path to the saved model weights that will be loaded into `model`.
        eval_path (str): Path to the saved evaluation dataset.
        verbose (bool, optional): True if `evaluate_sequences()` should print the F1, Precision, and Recall for each class. Defaults to True.

    Returns:
        1. Multi-class results (Mapping[int, Mapping[str, float]]): first map has keys as the classes (lemma indices) and value is 
                                                                    another map with key of "f1", "precision", or "recall" with corresponding values.
        2. Confusion Matrix (Mapping[int, Mapping[int, int]]): A confusion matrix with keys equal to the index of the gold tag, and a value of the 
                                                               map with the key as the predicted tag and corresponding count of that (gold, pred) pair.
        3. Accuracy (float): the total accuracy (num correct / total examples) across the evaluation set.
    """
    # load model
    model_state = torch.load(model_path)
    model.load_state_dict(model_state['params'])
    model.eval()  # set to eval mode

    # load in eval data 
    label_decoder = model_state['label_decoder']
    text_batches, index_batches, label_batches, _, label_decoder = utils.load_dataset(eval_path, label_decoder=label_decoder)
    
    logging.info(f"Evaluating model from {model_path} on evaluation file {eval_path}")

    correct = 0
    gold_tags, pred_tags = [label_batches], []
    GLOVE = get_glove(model.embedding_dim)
    # run eval on each example from dataset
    for sentence, pos_index, label in tqdm(zip(text_batches, index_batches, label_batches), "Evaluating examples from data file"):
        # tokenize raw text sentence using model
        # TODO: See if John approves of this fix
        tokenized_sentence = [GLOVE.stoi[word.lower()] if word.lower() in GLOVE.stoi else UNKNOWN_TOKEN_IDX for word in sentence]  # handle unknown tokens by randomizing their embedding

        pred = model_predict(model, tokenized_sentence, pos_index, sentence)
        correct += 1 if pred == label else 0 
        pred_tags += [pred]

    logging.info("Finished evaluating on dataset. Computing scores...")
    accuracy = correct / len(label_batches)
    mc_results, confusion = evaluate_sequences(gold_tags, [pred_tags], verbose=verbose)  
    # add brackets around batches of gold and pred tags because each batch is an element within the sequences in this helper
    if verbose:
        logging.info(f"Accuracy: {accuracy} ({correct}/{len(label_batches)})")
    
    return mc_results, confusion, accuracy


def transformer_pred(model: LemmaClassifierWithTransformer, text: List[str], pos_idx: int):
    """
    A LemmaClassifierWithTransformer is used to predict on a single text example, given the position index of the target token.

    Args:
        model (LemmaClassifierWithTransformer): A trained LemmaClassifierWithTransformer that is able to predict on a target token.
        text (List[str]): A sentence of words with each word as its own element.
        position_idx (int): The (zero-indexed) position of the target token in `text`.
    
    Returns:
        (int): The index of the predicted class in `model`'s output.
    """
    assert len(text) != 0, f"Text arg is empty. Please provide a proper input for model evaluation."
    if not isinstance(text[0], str):
        raise TypeError(f"Text variable must contain tokenized version of sentence, but instead found type {type(text[0])}.")
    
    with torch.no_grad():
        logits = model(text, pos_idx)
        predicted_class = torch.argmax(logits).item()
    return predicted_class


def evaluate_transformer(model:LemmaClassifierWithTransformer, model_path: str, eval_path: str, verbose: bool = True):
    """
    Helper function for transformer-model evaluation

    Args:
        model (LemmaClassifierWithTransformer): An instance of the LemmaClassifierWithTransformer class that has architecture initialized which
                                                 matches the model saved in `model_path`.
        model_path (str): Path to the saved model weights that will be loaded into `model`.
        eval_path (str): Path to the saved evaluation dataset.
        verbose (bool, optional): True if `evaluate_sequences()` should print the F1, Precision, and Recall for each class. Defaults to True.

    Returns:
        1. Multi-class results (Mapping[int, Mapping[str, float]]): first map has keys as the classes (lemma indices) and value is 
                                                                    another map with key of "f1", "precision", or "recall" with corresponding values.
        2. Confusion Matrix (Mapping[int, Mapping[int, int]]): A confusion matrix with keys equal to the index of the gold tag, and a value of the 
                                                               map with the key as the predicted tag and corresponding count of that (gold, pred) pair.
        3. Accuracy (float): the total accuracy (num correct / total examples) across the evaluation set.
    """
    # load model
    # TODO: need to save the label_decoder in the model file for the transformer version
    model_state = torch.load(model_path)
    model.load_state_dict(model_state['params'])
    model.eval()  # set to eval mode

    # load in eval data 
    label_decoder = model_state['label_decoder']
    text_batches, index_batches, label_batches, _, label_decoder = utils.load_dataset(eval_path, label_decoder=label_decoder)
    
    logging.info(f"Evaluating model from {model_path} on evaluation file {eval_path}")

    correct = 0
    gold_tags, pred_tags = [label_batches], []
    # run eval on each example from dataset
    for sentence, pos_index, label in tqdm(zip(text_batches, index_batches, label_batches), "Evaluating examples from data file"):
        pred = transformer_pred(model, sentence, pos_index)
        correct += 1 if pred == label else 0 
        pred_tags += [pred]

    logging.info("Finished evaluating on dataset. Computing scores...")
    accuracy = correct / len(label_batches)
    mc_results, confusion = evaluate_sequences(gold_tags, [pred_tags], verbose=verbose)  
    # add brackets around batches of gold and pred tags because each batch is an element within the sequences in this helper
    if verbose:
        logging.info(f"Accuracy: {accuracy} ({correct}/{len(label_batches)})")
    
    return mc_results, confusion, accuracy


def main(args=None):

    # TODO: can unify this script with train_model.py?
    # TODO: can save the model type in the model .pt, then
    # automatically figure out what type of model we are using by
    # looking in the file
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000, help="Number of tokens in vocab")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Number of dimensions in word embeddings (currently using GloVe)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Size of hidden layer")
    parser.add_argument("--output_dim", type=int, default=2, help="Size of output layer (number of classes)")
    parser.add_argument("--charlm", action='store_true', default=False, help="Whether not to use the charlm embeddings")
    parser.add_argument('--charlm_shorthand', type=str, default=None, help="Shorthand for character-level language model training corpus.")
    parser.add_argument("--charlm_forward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_forward.pt"), help="Path to forward charlm file")
    parser.add_argument("--charlm_backward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_backwards.pt"), help="Path to backward charlm file")
    parser.add_argument("--save_name", type=str, default=os.path.join(os.path.dirname(__file__), "saved_models", "lemma_classifier_model.pt"), help="Path to model save file")
    parser.add_argument("--model_type", type=str, default="roberta", help="Which transformer to use ('bert' or 'roberta' or 'lstm')")
    parser.add_argument("--bert_model", type=str, default=None, help="Use a specific transformer instead of the default bert/roberta")
    parser.add_argument("--eval_file", type=str, help="path to evaluation file")

    args = parser.parse_args(args)

    logging.info("Running training script with the following args:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("------------------------------------------------------------")

    vocab_size = args.vocab_size
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    use_charlm = args.charlm
    forward_charlm_file = args.charlm_forward_file
    backward_charlm_file = args.charlm_backward_file
    save_name = args.save_name 
    model_type = args.model_type
    eval_path = args.eval_file

    if model_type.lower() == "lstm" and use_charlm:
        # Evaluate charlm
        model = LemmaClassifier(vocab_size=vocab_size,
                                embedding_dim=embedding_dim,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,
                                embeddings=get_glove(embedding_dim).vectors,
                                charlm=True,
                                charlm_forward_file=forward_charlm_file,
                                charlm_backward_file=backward_charlm_file)
    elif model_type.lower() == "lstm" and not use_charlm:
        # Evaluate standard model (bi-LSTM with GloVe embeddings, no charlm)
        model = LemmaClassifier(vocab_size=vocab_size,
                                embedding_dim=embedding_dim,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,
                                embeddings=get_glove(embedding_dim).vectors,
                                )
    elif model_type.lower() == "roberta":
        # Evaluate Transformer (BERT or ROBERTA)
        model = LemmaClassifierWithTransformer(output_dim=output_dim, transformer_name="roberta-base")
    elif model_type.lower() == "bert":
        # Evaluate Transformer (BERT or ROBERTA)
        model = LemmaClassifierWithTransformer(output_dim=output_dim, transformer_name="bert-base-uncased")
    elif model_type.lower() == "transformer":
        model = LemmaClassifierWithTransformer(output_dim=output_dim, transformer_name=args.bert_model)
    else:
        raise ValueError("Unknown model type %s" % model_type)

    logging.info(f"Attempting evaluation of model from {save_name} on file {eval_path}")

    if model_type.lower() == "lstm":
        # for LSTM models
        mcc_results, confusion, acc = evaluate_model(model, save_name, eval_path)

    elif model_type.lower() == "roberta" or model_type.lower() == "bert" or model_type.lower() == "transformer":
        # for transformer
        mcc_results, confusion, acc = evaluate_transformer(model, save_name, eval_path)

    logging.info(f"MCC Results: {dict(mcc_results)}")
    logging.info("______________________________________________")
    logging.info(f"Confusion: {dict(confusion)}")
    logging.info("______________________________________________")
    logging.info(f"Accuracy: {acc}")

    

if __name__ == "__main__":
    main()
