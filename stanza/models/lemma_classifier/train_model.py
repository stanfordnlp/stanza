"""
The code in this file works to train a lemma classifier for 's
"""

import argparse
import logging
import os 

import torch 
import torch.nn as nn

from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.lemma_classifier.base_trainer import BaseLemmaClassifierTrainer
from stanza.models.lemma_classifier.constants import DEFAULT_BATCH_SIZE
from stanza.models.lemma_classifier.model import LemmaClassifierLSTM

logger = logging.getLogger('stanza.lemmaclassifier')

class LemmaClassifierTrainer(BaseLemmaClassifierTrainer):
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
        super().__init__()

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
            logger.debug("Using CE loss")
        elif loss_func == "weighted_bce":
            self.criterion = nn.BCEWithLogitsLoss()
            self.weighted_loss = True  # used to add weights during train time.
            logger.debug("Using Weighted BCE loss")
        else:
            raise ValueError("Must enter a valid loss function (e.g. 'ce' or 'weighted_bce')")

    def build_model(self, label_decoder, upos_to_id):
        return LemmaClassifierLSTM(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim, self.vocab_map, self.embeddings, label_decoder,
                                   charlm=self.use_charlm, charlm_forward_file=self.forward_charlm_file, charlm_backward_file=self.backward_charlm_file,
                                   upos_emb_dim=self.upos_emb_dim, num_heads=self.num_heads, upos_to_id=upos_to_id)

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
    parser.add_argument("--save_name", type=str, default=os.path.join(os.path.dirname(__file__), "saved_models", "lemma_classifier_model_weighted_loss_charlm_new.pt"), help="Path to model save file")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of examples to include in each batch")
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
    num_epochs = args.num_epochs
    train_file = args.train_file
    weighted_loss = args.weighted_loss
    eval_file = args.eval_file

    args = vars(args)

    if os.path.exists(save_name):
        raise FileExistsError(f"Save name {save_name} already exists. Training would override existing data. Aborting...")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file {train_file} not found. Try again with a valid path.")

    logger.info("Running training script with the following args:")
    for arg in args:
        logger.info(f"{arg}: {args[arg]}")
    logger.info("------------------------------------------------------------")

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
        num_epochs=num_epochs, save_name=save_name, args=args, eval_file=eval_file, train_file=train_file
    )

    return trainer

if __name__ == "__main__":
    main()

