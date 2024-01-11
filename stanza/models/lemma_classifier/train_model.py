"""
The code in this file works to train a lemma classifier for 's
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import os 
import logging
import argparse
from os import path
from os import remove
from typing import List, Tuple, Any, Mapping

from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.common.utils import default_device
from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.constants import ModelType, DEFAULT_BATCH_SIZE
from stanza.models.lemma_classifier.model import LemmaClassifierLSTM
from stanza.utils.get_tqdm import get_tqdm
from stanza.models.lemma_classifier.evaluate_models import evaluate_model

tqdm = get_tqdm()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LemmaClassifierTrainer():
    """
    Class to assist with training a LemmaClassifierLSTM
    """

    def __init__(self, embedding_file: str, hidden_dim: int, use_charlm: bool = False, forward_charlm_file: str = None, backward_charlm_file: str = None, upos_emb_dim: int = 20, num_heads: int = 0, lr: float = 0.001, loss_func: str = None, eval_file: str = None):
        """
        Initializes the LemmaClassifierTrainer class.
        
        Args:
            embedding_file (str): What word embeddings file to use.  Use a Stanza pretrain .pt
            hidden_dim (int): Size of hidden vectors in LSTM layers
            use_charlm (bool, optional): Whether to use charlm embeddings as well. Defaults to False.
            eval_file (str): File used as dev set to evaluate which model gets saved
            forward_charlm_file (str): Path to the forward pass embeddings for the charlm 
            backward_charlm_file (str): Path to the backward pass embeddings for the charlm
            upos_emb_dim (int): The dimension size of UPOS tag embeddings
            num_heads (int): The number of attention heads to use.
            lr (float): Learning rate, defaults to 0.001.
            loss_func (str): Which loss function to use (either 'ce' or 'weighted_bce') 

        Raises:
            FileNotFoundError: If the forward charlm file is not present
            FileNotFoundError: If the backward charlm file is not present
        """
        self.hidden_dim = hidden_dim
        self.upos_emb_dim = upos_emb_dim

        # Load word embeddings
        pt = load_pretrain(embedding_file)
        emb_matrix = pt.emb
        # TODO: could refactor only the trained embeddings, then turn freezing back on, then don't save the full PT with the model
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=False)
        self.embeddings.weight.requires_grad = True
        self.vocab_map = { word.replace('\xa0', ' '): i for i, word in enumerate(pt.vocab) }
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        # Load CharLM embeddings
        if use_charlm and forward_charlm_file is not None and not os.path.exists(forward_charlm_file):
            raise FileNotFoundError(f"Could not find forward charlm file: {forward_charlm_file}")
        if use_charlm and backward_charlm_file is not None and not os.path.exists(backward_charlm_file):
            raise FileNotFoundError(f"Could not find backward charlm file: {backward_charlm_file}")

        # TODO: just pass around the args instead
        self.use_charlm = use_charlm
        self.forward_charlm_file = forward_charlm_file
        self.backward_charlm_file = backward_charlm_file
        self.num_heads = num_heads
        self.lr = lr

        # Find loss function
        if loss_func == "ce":
            self.criterion = nn.CrossEntropyLoss()
            self.weighted_loss = False
            logging.debug("Using CE loss")
        elif loss_func == "weighted_bce":
            self.criterion = nn.BCEWithLogitsLoss()
            self.weighted_loss = True  # used to add weights during train time.
            logging.debug("Using Weighted BCE loss")
        else:
            raise ValueError("Must enter a valid loss function (e.g. 'ce' or 'weighted_bce')")

    def configure_weighted_loss(self, label_decoder: Mapping, counts: Mapping):
        """
        If applicable, this function will update the loss function of the LemmaClassifierLSTM model to become BCEWithLogitsLoss.
        The weights are determined by the counts of the classes in the dataset. The weights are inversely proportional to the
        frequency of the class in the set. E.g. classes with lower frequency will have higher weight.
        """
        weights = [0 for _ in label_decoder.keys()]  # each key in the label decoder is one class, we have one weight per class
        total_samples = sum(counts.values())
        for class_idx in counts:
            weights[class_idx] = total_samples / (counts[class_idx] * len(counts))  # weight_i = total / (# examples in class i * num classes)
        weights = torch.tensor(weights)
        logging.info(f"Using weights {weights} for weighted loss.")
        self.criterion = nn.BCEWithLogitsLoss(weight=weights)

    def train(self, num_epochs: int, save_name: str, args: Mapping, eval_file: str, **kwargs) -> None:

        """
        Trains a model on batches of texts, position indices of the target token, and labels (lemma annotation) for the target token.

        Args:
            texts_batch (List[List[str]]): Batches of tokenized texts, one per sentence. Expected to contain at least one instance of the target token.
            positions_batch (List[int]): Batches of position indices (zero-indexed) for the target token, one per input sentence. 
            labels_batch (List[int]): Batches of labels for the target token, one per input sentence. 
            num_epochs (int): Number of training epochs
            save_name (str): Path to file where trained model should be saved. 

        Kwargs:
            train_path (str): Path to data file, containing tokenized text sentences, token index and true label for token lemma on each line.      
            batch_size (int): Number of examples to include in each batch.   
        """
        
        device = default_device() # Put model on GPU (if possible)

        train_path = kwargs.get("train_path")
        upos_to_id = {}
        if train_path:  # use file to train model
            text_batches, idx_batches, upos_batches, label_batches, counts, label_decoder, upos_to_id = utils.load_dataset(train_path, get_counts=self.weighted_loss, batch_size=kwargs.get("batch_size", DEFAULT_BATCH_SIZE)) 
            self.output_dim = len(label_decoder)
            logging.info(f"Loaded dataset successfully from {train_path}")
            logging.info(f"Using label decoder: {label_decoder}  Output dimension: {self.output_dim}")

        self.model = LemmaClassifierLSTM(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim, self.vocab_map, self.embeddings, label_decoder,
                                         charlm=self.use_charlm, charlm_forward_file=self.forward_charlm_file, charlm_backward_file=self.backward_charlm_file,
                                         upos_emb_dim=self.upos_emb_dim, num_heads=self.num_heads, upos_to_id=upos_to_id)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.to(device)
        logging.info(f"Device chosen: {device}. {next(self.model.parameters()).device}")

        assert len(text_batches) == len(idx_batches) == len(label_batches), f"Input batch sizes did not match ({len(text_batches)}, {len(idx_batches)}, {len(label_batches)})."
        if path.exists(save_name):
            raise FileExistsError(f"Save name {save_name} already exists; training would overwrite previous file contents. Aborting...")
        
        if self.weighted_loss:
            self.configure_weighted_loss(label_decoder, counts)

        # Put the criterion on GPU too
        logging.debug(f"Criterion on {next(self.model.parameters()).device}")
        self.criterion = self.criterion.to(next(self.model.parameters()).device)

        best_model, best_f1 = None, float("-inf")  # Used for saving checkpoints of the model
        logging.info("Embedding norm: %s", torch.linalg.norm(self.model.embedding.weight))
        for epoch in range(num_epochs):
            # go over entire dataset with each epoch
            for texts, positions, upos_tags, labels in tqdm(zip(text_batches, idx_batches, upos_batches, label_batches), total=len(text_batches)):  

                self.optimizer.zero_grad()
                output = self.model(positions, texts, upos_tags)

                # Compute loss, which is different if using CE or BCEWithLogitsLoss
                if self.weighted_loss:  # BCEWithLogitsLoss requires a vector for target where probability is 1 on the true label class, and 0 on others.
                    # TODO: three classes?
                    targets = torch.stack([torch.tensor([1, 0]) if label == 0 else torch.tensor([0, 1]) for label in labels]).to(dtype=torch.float32).to(device)
                    # should be shape size (batch_size, 2)

                else:  # CELoss accepts target as just raw label
                    targets = labels.to(device)

                loss = self.criterion(output, targets)

                loss.backward()
                self.optimizer.step()

            if eval_file:
                _, _, _, f1 = evaluate_model(self.model, eval_file, is_training=True)
                logging.info(f"Weighted f1 for model: {f1}")
                if f1 > best_f1:
                    best_f1 = f1
                    self.model.save(save_name, args)
                    logging.info(f"New best model: weighted f1 score of {f1}.")
            else:
                self.model.save(save_name, args)

            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


def build_argparse():
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
    parser.add_argument("--save_name", type=str, default=path.join(path.dirname(__file__), "saved_models", "lemma_classifier_model_weighted_loss_charlm_new.pt"), help="Path to model save file")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of examples to include in each batch")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(__file__), "data", "processed_ud_en", "combined_train.txt"), help="Full path to training file")
    parser.add_argument("--weighted_loss", action='store_true', dest='weighted_loss', default=False, help="Whether to use weighted loss during training.")
    parser.add_argument("--eval_file", type=str, default=os.path.join(os.path.dirname(__file__), "data", "processed_ud_en", "combined_dev.txt"), help="Path to dev file used to evaluate model for saves")
    return parser

def main(args=None):
    parser = build_argparse()
    args = parser.parse_args(args)

    hidden_dim = args.hidden_dim
    wordvec_pretrain_file = args.wordvec_pretrain_file
    use_charlm = args.use_charlm
    forward_charlm_file = args.charlm_forward_file
    backward_charlm_file = args.charlm_backward_file
    upos_emb_dim = args.upos_emb_dim
    use_attention = args.attn 
    num_heads = args.num_heads
    save_name = args.save_name 
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train_file = args.train_file
    weighted_loss = args.weighted_loss
    eval_file = args.eval_file

    args = vars(args)

    if os.path.exists(save_name):
        raise FileExistsError(f"Save name {save_name} already exists. Training would override existing data. Aborting...")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file {train_file} not found. Try again with a valid path.")

    logging.info("Running training script with the following args:")
    for arg in args:
        logging.info(f"{arg}: {args[arg]}")
    logging.info("------------------------------------------------------------")

    trainer = LemmaClassifierTrainer(embedding_file=wordvec_pretrain_file,
                                     hidden_dim=hidden_dim,
                                     use_charlm=use_charlm,
                                     forward_charlm_file=forward_charlm_file,
                                     backward_charlm_file=backward_charlm_file,
                                     upos_emb_dim=upos_emb_dim,
                                     num_heads=num_heads if use_attention else 0,
                                     lr=lr,
                                     loss_func="weighted_bce" if weighted_loss else "ce",
                                     )

    trainer.train(
        num_epochs=num_epochs, save_name=save_name, args=args, eval_file=eval_file, train_path=train_file, batch_size=batch_size
    )

    return trainer

if __name__ == "__main__":
    main()

