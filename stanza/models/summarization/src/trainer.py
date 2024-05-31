import argparse
import logging
import os 
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer

# To add Stanza modules, TODO remove this and just EXPORT this to the sys path manually before running
ROOT = '/Users/alexshan/Desktop/stanza'
sys.path.append(ROOT)

from stanza.models.common.utils import default_device
from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.summarization.constants import * 
from stanza.models.summarization.src.model import *
from stanza.utils.get_tqdm import get_tqdm
from stanza.models.summarization.src.utils import *
from stanza.models.summarization.src.prepare_dataset import Dataset
from stanza.models.summarization.src.evaluate import evaluate_from_path

from typing import List, Tuple, Any, Mapping

torch.set_printoptions(threshold=100, edgeitems=5, linewidth=100)
logger = logging.getLogger('stanza.summarization') 
logger.propagate = False

# Check if the logger has handlers already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

tqdm = get_tqdm()


class SummarizationTrainer():

    def __init__(self, model_args: dict, embedding_file: str, lr: float):
        """
        Model arguments:
        {
        batch_size (int): size of data batches used during training, 
        enc_hidden_dim (int): Size of encoder hidden state,
        enc_num_layers (int): Number of layers in the encoder LSTM,
        dec_hidden_dim (int): Size of decoder hidden state,
        dec_num_layers (int): Number of layers in the decoder LSTM,
        pgen (bool): Whether to use the pointergen feature in the model,
        coverage (bool): Whether to include coverage vectors in the decoder,
        }

        embedding_file (str): Path to the word vector pretrain file for embedding layer
        lr (float): Learning rate during training
        """
        self.model_args = model_args

        pt = load_pretrain(embedding_file)
        self.pt_embedding = pt
        self.lr = lr
        self.device = default_device() 
        self.max_enc_steps = self.model_args.get("max_enc_steps", None)
        self.max_dec_steps = self.model_args.get("max_dec_steps", None)

    def build_model(self) -> BaselineSeq2Seq:
        """
        Build the model for training using the model args

        Raises any errors depending on model argument errors
        """

        # parse input for valid args
        batch_size = self.model_args.get("batch_size", DEFAULT_BATCH_SIZE)
        encoder_hidden_dim = self.model_args.get("enc_hidden_dim", DEFAULT_ENCODER_HIDDEN_DIM)
        encoder_num_layers = self.model_args.get("enc_num_layers", DEFAULT_ENCODER_NUM_LAYERS)
        decoder_hidden_dim = self.model_args.get("dec_hidden_dim", DEFAULT_DECODER_HIDDEN_DIM)
        decoder_num_layers = self.model_args.get("dec_num_layers", DEFAULT_DECODER_NUM_LAYERS)
        pgen = self.model_args.get("pgen", False)
        coverage = self.model_args.get("coverage", False)
        use_charlm = self.model_args.get("charlm", False)
        charlm_forward_file = self.model_args.get("charlm_forward_file", None)
        charlm_backward_file = self.model_args.get("charlm_backward_file", None)
        max_enc_steps = self.model_args.get("max_enc_steps", None)
        max_dec_steps = self.model_args.get("max_dec_steps", None)

        parsed_model_args = {
            "batch_size": batch_size,
            "encoder_hidden_dim": encoder_hidden_dim,
            "encoder_num_layers": encoder_num_layers,
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_num_layers": decoder_num_layers,
            "pgen": pgen,
            "coverage": coverage,
            "max_enc_steps": max_enc_steps,
            "max_dec_steps": max_dec_steps,
        }

        # return the model obj
        return BaselineSeq2Seq(parsed_model_args, self.pt_embedding, device=self.device, 
                               use_charlm=use_charlm, charlm_forward_file=charlm_forward_file, charlm_backward_file=charlm_backward_file)

    def train(self, num_epochs: int, save_name: str, train_file: str, eval_file: str) -> None:
        """
        Trains a model on batches of texts

        Args:
            num_epochs (int): Number of training epochs 
            save_name (str): Path to store trained model
            eval_file (str): Path to the validation set file roots evaluating model checkpoints
            train_file (str): Path to training data root containing chunked files with tokenized text for each article + summary

        Returns:
            None (model with best validation set performance will be saved to the save file)
        """
        best_rouge = 0
        best_model_path = os.path.join(os.path.dirname(save_name), "temp_best")  # best checkpoint so far
        device = default_device()
        # Load model in
        self.model = self.build_model()

        self.model.to(device)

        # Get dataset 
        batch_size = self.model_args.get("batch_size", DEFAULT_BATCH_SIZE)
        dataset = Dataset(train_file, batch_size)

        # Load optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.NLLLoss(reduction="none")
        self.criterion = self.criterion.to(next(self.model.parameters()).device)

        for epoch in range(num_epochs):
            # Iterate through dataset  TODO fix this
            # for batch in tokenized_dataset:
            for articles, summaries in dataset:

                # Get model output
                self.optimizer.zero_grad()
                output, attention_scores, coverage_vectors = self.model(articles, summaries)  # (batch size, seq len, vocab size)
                output = output.permute(0, 2, 1)   # (batch size, vocab size, seq len)

                target_indices = convert_text_to_token_ids(self.model.vocab_map, summaries, UNK_ID, self.max_dec_steps)  # does this need to use the extended vocab map? If so, how? Do we have to compute it beforehand here?
                # target_indices = self.model.build_extended_vocab_map(summaries)
                """
                Do we use the extended vocab map to get the target indices? 

                Reasons for yes: 
                - We need to get the indices on the reference summary for OOV words, as they may appear in there
                - If we just use the vocab map, any reference summary words may just have nothing

                Reasons for no:
                - Are we adding too many indices?

                For a reference summary, it might contain words not in the vocab or the original article. 
                What happens in this case? 
                - We create a target index for it, and it goes into the extended vocab
                - The extended vocab was initially meant for input text words that are new

                - Ultimately why would this be an issue? We give a new index if its OOV, and we train it to include that sometimes.

                If we were using a normal vocab map, there wouldn't be any loss for a word that is OOV in the reference summary.
                That means during loss computation, it would expect token UNK when we predict within our vocab (which contains unk!)


                """

                print("TARGET INDICES SHAPE",target_indices.shape)
                # Compute losses (base loss)
                log_loss = self.criterion(output, target_indices)
                # coverage loss
                if self.model.coverage:
                    coverage_losses = torch.sum(torch.min(attention_scores, coverage_vectors), dim=-1)

                    combined_losses = log_loss + coverage_losses

                    print(f"LOG LOSS: {log_loss}   {log_loss.shape}")  # should be (batch size, seq len)
                    print(f"COV LOSS {coverage_losses}    {coverage_losses.shape}")  # (batch size, seq len)
                else:
                    combined_losses = log_loss 
                
                # backwards
                sequence_loss = combined_losses.mean(dim=1)

                print(f"sequence loss shape {sequence_loss.shape}")  # (batch size)
                batch_loss = sequence_loss.mean()
                print("BATCH LOSS SHAPE", batch_loss.shape, batch_loss)  # ([])
                batch_loss.backward()
                self.optimizer.step()
            # TODO evaluate model checkpoint on val set
            torch.save(self.model, save_name)
            results = evaluate_from_path(
                                        model_path=save_name,
                                        eval_path=eval_file,
                                        logger=logger
                                        )
        # compare to best checkpoint
        if results.get('rougeLsum') > best_rouge:
            best_rouge = results.get("rougeLsum")
            torch.save(self.model, best_model_path)
        best_model = torch.load(best_model_path)
        torch.save(best_model, save_name)


def parse_args():
    parser = argparse.ArgumentParser()
    # Model args
    parser.add_argument("--enc_hidden_dim", type=int, default=DEFAULT_ENCODER_HIDDEN_DIM, help="Size of encoder hidden states")
    parser.add_argument("--enc_num_layers", type=int, default=DEFAULT_ENCODER_NUM_LAYERS, help="Number of layers in the encoder LSTM")
    parser.add_argument("--dec_hidden_dim", type=int, default=DEFAULT_DECODER_HIDDEN_DIM, help="Size of decoder hidden state vector")
    parser.add_argument("--dec_num_layers", type=int, default=DEFAULT_DECODER_NUM_LAYERS, help="Number of layers in the decoder LSTM")
    parser.add_argument("--pgen", action="store_true", dest="pgen", default=False, help="Use pointergen probabilities to point to input text")
    parser.add_argument("--coverage", action="store_true", dest="coverage", default=False, help="Use coverage vectors during decoding stage")
    # Training args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for data processing")
    parser.add_argument("--save_name", type=str, default=DEFAULT_SAVE_NAME, help="Path to destination for final trained model.")
    parser.add_argument("--eval_path", type=str, default=DEFAULT_EVAL_ROOT, help="Path to the validation set root")
    parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN_ROOT, help="Path to the training data root")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wordvec_pretrain_file", type=str, default=DEFAULT_WORDVEC_PRETRAIN_FILE, help="Path to pretrained word embeddings file")
    parser.add_argument("--charlm", action="store_true", dest="charlm", default=False, help="Use character language model embeddings.")
    parser.add_argument("--charlm_forward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_forward.pt"), help="Path to forward charlm file")
    parser.add_argument("--charlm_backward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_backwards.pt"), help="Path to backward charlm file")
    parser.add_argument("--max_enc_steps", type=int, default=None, help="Limit on article sizes (will be truncated)")
    parser.add_argument("--max_dec_steps", type=int, default=None, help="Limit on summary sizes (will be truncated)")
    return parser

def main():
    argparser = parse_args()
    args = argparser.parse_args()

    enc_hidden_dim = args.enc_hidden_dim
    enc_num_layers = args.enc_num_layers
    dec_hidden_dim = args.dec_hidden_dim
    dec_num_layers = args.dec_num_layers
    pgen = args.pgen
    coverage = args.pgen

    batch_size = args.batch_size
    save_name = args.save_name
    eval_path = args.eval_path
    train_path = args.train_path
    num_epochs = args.num_epochs
    lr = args.lr
    wordvec_pretrain_file = args.wordvec_pretrain_file
    charlm_forward_file = args.charlm_forward_file
    charlm_backward_file = args.charlm_backward_file
    use_charlm = args.charlm
    max_enc_steps = args.max_enc_steps
    max_dec_steps = args.max_dec_steps

    if not os.path.exists(eval_path):
        no_eval_file_msg = f"Could not find provided eval dir: {eval_path}"
        logger.error(no_eval_file_msg)
        raise FileNotFoundError(no_eval_file_msg)
    if not os.path.exists(train_path):
        no_train_file_msg = f"Could not find provided train dir: {train_path}"
        logger.error(no_train_file_msg)
        raise FileNotFoundError(no_train_file_msg)
    if not os.path.exists(wordvec_pretrain_file):
        no_wordvec_file_msg = f"Could not find provided wordvec pretrain file {wordvec_pretrain_file}"
        logger.error(no_wordvec_file_msg)
        raise FileNotFoundError(no_wordvec_file_msg)
    if use_charlm:
        if not os.path.exists(charlm_forward_file):
            no_charlm_forward_file_msg = f"Could not find provided charlm forward file {charlm_forward_file}"
            logger.error(no_charlm_forward_file_msg)
            raise FileNotFoundError(no_charlm_forward_file_msg)
        if not os.path.exists(charlm_backward_file):
            no_charlm_backward_file_msg = f"Could not find provided charlm backward file {charlm_backward_file}"
            logger.error(no_charlm_backward_file_msg)
            raise FileNotFoundError(no_charlm_backward_file_msg)
    
    args = vars(args)
    logger.info("Using the following args for training:")
    for arg, val in args.items():
        logger.info(f"{arg}: {val}")

    trainer = SummarizationTrainer(
        model_args=args,
        embedding_file=wordvec_pretrain_file,
        lr=lr
    )
    trainer.train(
        num_epochs=num_epochs,
        save_name=save_name,
        train_file=train_path,
        eval_file=eval_path
    )


if __name__ == "__main__":
    main()