"""
Utils to evaluate many models of the same type at once
"""
import argparse
import os
import logging

from stanza.models.lemma_classifier.evaluate_models import main as evaluate_main


logger = logging.getLogger('stanza.lemmaclassifier')

def evaluate_n_models(path_to_models_dir, args):

    total_results = {
        "be": 0.0,
        "have": 0.0,
        "accuracy": 0.0,
        "weighted_f1": 0.0
    }
    paths = os.listdir(path_to_models_dir)
    num_models = len(paths)
    for model_path in paths:
        full_path = os.path.join(path_to_models_dir, model_path)
        args.save_name = full_path
        mcc_results, confusion, acc, weighted_f1 = evaluate_main(predefined_args=args)

        for lemma in mcc_results:

            lemma_f1 = mcc_results.get(lemma, None).get("f1") * 100
            total_results[lemma] += lemma_f1

        total_results["accuracy"] += acc
        total_results["weighted_f1"] += weighted_f1

    total_results["be"] /= num_models
    total_results["have"] /= num_models
    total_results["accuracy"] /= num_models
    total_results["weighted_f1"] /= num_models

    logger.info(f"Models in {path_to_models_dir} had average weighted f1 of {100 * total_results['weighted_f1']}.\nLemma 'be' had f1: {total_results['be']}\nLemma 'have' had f1: {total_results['have']}.\nAccuracy: {100 * total_results['accuracy']}.\n ({num_models} models evaluated).")
    return total_results


def main():
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

    # Args specific to several model eval
    parser.add_argument("--base_path", type=str, default=None, help="path to dir for eval")

    args = parser.parse_args()
    evaluate_n_models(args.base_path, args)


if __name__ == "__main__":
    main()
