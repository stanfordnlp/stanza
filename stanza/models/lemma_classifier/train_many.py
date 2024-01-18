"""
Utils for training and evaluating multiple models simultaneously
"""

import argparse
import os

from stanza.models.lemma_classifier.train_lstm_model import main as train_lstm_main
from stanza.models.lemma_classifier.train_transformer_model import main as train_tfmr_main
from stanza.models.lemma_classifier.constants import DEFAULT_BATCH_SIZE


change_params_map = {
    "lstm_layer": [16, 32, 64, 128, 256, 512],
    "upos_emb_dim": [5, 10, 20, 30],
    "training_size": [150, 300, 450, 600, 'full'],
}  # TODO: Add attention

def train_n_models(num_models: int, base_path: str, args):

    if args.change_param == "lstm_layer":
        for num_layers in change_params_map.get("lstm_layer", None):
            for i in range(num_models):
                new_save_name = os.path.join(base_path, f"{num_layers}_{i}.pt")
                args.save_name = new_save_name
                args.hidden_dim = num_layers
                train_lstm_main(predefined_args=args)

    if args.change_param == "upos_emb_dim":
        for upos_dim in change_params_map("upos_emb_dim", None):
            for i in range(num_models):
                new_save_name = os.path.join(base_path, f"dim_{upos_dim}_{i}.pt")
                args.save_name = new_save_name
                args.upos_emb_dim = upos_dim
                train_lstm_main(predefined_args=args)

    if args.change_param == "training_size":
        for size in change_params_map.get("training_size", None):
            for i in range(num_models):
                new_save_name = os.path.join(base_path, f"{size}_examples_{i}.pt")
                new_train_file = os.path.join(os.path.dirname(__file__), "data", "processed_ud_en", "combined_train.txt")
                args.save_name = new_save_name
                args.train_file = new_train_file
                train_lstm_main(predefined_args=args)

    if args.change_param == "base":
        for i in range(num_models):
            new_save_name = os.path.join(base_path, f"lstm_model_{i}.pt")
            args.save_name = new_save_name
            args.weighted_loss = False
            train_lstm_main(predefined_args=args)

            if not args.weighted_loss:
                args.weighted_loss = True
                new_save_name = os.path.join(base_path, f"lstm_model_wloss_{i}.pt")
                args.save_name = new_save_name
                train_lstm_main(predefined_args=args)

    if args.change_param == "base_charlm":
        for i in range(num_models):
            new_save_name = os.path.join(base_path, f"lstm_charlm_{i}.pt")
            args.save_name = new_save_name
            train_lstm_main(predefined_args=args)

    if args.change_param == "base_charlm_upos":
        for i in range(num_models):
            new_save_name = os.path.join(base_path, f"lstm_charlm_upos_{i}.pt")
            args.save_name = new_save_name
            train_lstm_main(predefined_args=args)

    if args.change_param == "base_upos":
        for i in range(num_models):
            new_save_name = os.path.join(base_path, f"lstm_upos_{i}.pt")
            args.save_name = new_save_name
            train_lstm_main(predefined_args=args)

    if args.change_param == "attn_model":
        for i in range(num_models):
            new_save_name = os.path.join(base_path, f"attn_model_{args.num_heads}_heads_{i}.pt")
            args.save_name = new_save_name
            train_lstm_main(predefined_args=args)

def train_n_tfmrs(num_models: int, base_path: str, args):

    if args.multi_train_type == "tfmr":

        for i in range(num_models):

            if args.change_param == "bert":
                new_save_name = os.path.join(base_path, f"bert_{i}.pt")
                args.save_name = new_save_name
                args.loss_fn = "ce"
                train_tfmr_main(predefined_args=args)

                new_save_name = os.path.join(base_path, f"bert_wloss_{i}.pt")
                args.save_name = new_save_name
                args.loss_fn = "weighted_bce"
                train_tfmr_main(predefined_args=args)

            elif args.change_param == "roberta":
                new_save_name = os.path.join(base_path, f"roberta_{i}.pt")
                args.save_name = new_save_name
                args.loss_fn = "ce"
                train_tfmr_main(predefined_args=args)

                new_save_name = os.path.join(base_path, f"roberta_wloss_{i}.pt")
                args.save_name = new_save_name
                args.loss_fn = "weighted_bce"
                train_tfmr_main(predefined_args=args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=256, help="Size of hidden layer")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=os.path.join(os.path.dirname(__file__), "pretrain", "glove.pt"), help='Exact name of the pretrain file to read')
    parser.add_argument("--charlm", action='store_true', dest='use_charlm', default=False, help="Whether not to use the charlm embeddings")
    parser.add_argument('--charlm_shorthand', type=str, default=None, help="Shorthand for character-level language model training corpus.")
    parser.add_argument("--charlm_forward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_forward.pt"), help="Path to forward charlm file")
    parser.add_argument("--charlm_backward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_backwards.pt"), help="Path to backward charlm file")
    parser.add_argument("--upos_emb_dim", type=int, default=20, help="Dimension size for UPOS tag embeddings.")
    parser.add_argument("--use_attn", action='store_true', dest='attn', default=False, help='Whether to use multihead attention instead of LSTM.')
    parser.add_argument("--num_heads", type=int, default=0, help="Number of heads to use for multihead attention.")
    parser.add_argument("--save_name", type=str, default=os.path.join(os.path.dirname(__file__), "saved_models", "lemma_classifier_model_weighted_loss_charlm_new.pt"), help="Path to model save file")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of examples to include in each batch")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(__file__), "data", "processed_ud_en", "combined_train.txt"), help="Full path to training file")
    parser.add_argument("--weighted_loss", action='store_true', dest='weighted_loss', default=False, help="Whether to use weighted loss during training.")
    parser.add_argument("--eval_file", type=str, default=os.path.join(os.path.dirname(__file__), "data", "processed_ud_en", "combined_dev.txt"), help="Path to dev file used to evaluate model for saves")
    # Tfmr-specific args
    parser.add_argument("--model_type", type=str, default="roberta", help="Which transformer to use ('bert' or 'roberta')")
    parser.add_argument("--bert_model", type=str, default=None, help="Use a specific transformer instead of the default bert/roberta")
    parser.add_argument("--loss_fn", type=str, default="weighted_bce", help="Which loss function to train with (e.g. 'ce' or 'weighted_bce')")
    # Multi-model train args
    parser.add_argument("--multi_train_type", type=str, default="lstm", help="Whether you are attempting to multi-train an LSTM or transformer")
    parser.add_argument("--multi_train_count", type=int, default=5, help="Number of each model to build")
    parser.add_argument("--base_path", type=str, default=None, help="Path to start generating model type for.")
    parser.add_argument("--change_param", type=str, default=None, help="Which hyperparameter to change when training")


    args = parser.parse_args()

    if args.multi_train_type == "lstm":
        train_n_models(num_models=args.multi_train_count,
                       base_path=args.base_path,
                       args=args)
    elif args.multi_train_type == "tfmr":
        train_n_tfmrs(num_models=args.multi_train_count,
                      base_path=args.base_path,
                      args=args)
    else:
        raise ValueError(f"Improper input {args.multi_train_type}")

if __name__ == "__main__":
    main()
