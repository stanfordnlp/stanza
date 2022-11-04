import argparse
import copy
import logging
import math
import os
import random
import time
from types import SimpleNamespace
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import dataset

from torchtext.vocab import build_vocab_from_iterator

from stanza.models.common import utils

tqdm = utils.get_tqdm()

logger = logging.getLogger('stanza')

class TransformerModel(nn.Module):

    def __init__(self, vocab, args):
        super().__init__()
        self.model_type = 'Transformer'
        self.vocab = vocab

        self.config = SimpleNamespace(d_embedding = args.d_embedding,
                                      d_hid = args.d_hid,
                                      dropout = args.dropout,
                                      n_heads = args.n_heads,
                                      n_layers = args.n_layers)

        # keep the criterion here so we can use it for the scoring as part of the model
        # sum loss?  should penalize sentences which are too long for no reason
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.n_tokens = len(vocab)
        self.pos_encoder = PositionalEncoding(self.config.d_embedding, self.config.dropout)
        encoder_layers = TransformerEncoderLayer(self.config.d_embedding, self.config.n_heads, self.config.d_hid, self.config.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.config.n_layers)
        self.encoder = nn.Embedding(self.n_tokens, self.config.d_embedding)
        self.decoder = nn.Linear(self.config.d_embedding, self.n_tokens)

        # we assume text is always whitespace separated
        # that is the format of the trees used, after all
        self.tokenizer = lambda x: x.strip().split()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.config.d_embedding)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(output)
        return output

    def score(self, sentences, batch_size=10, use_tqdm=False):
        device = next(self.parameters()).device
        if isinstance(sentences, str):
            sentences = [sentences]
        if len(sentences) == 0:
            return []
        with torch.no_grad():
            data, indices = data_process(self.vocab, self.tokenizer, None, iter(sentences))
            data = batchify(data, batch_size, None)
            if use_tqdm:
                data = tqdm(data, leave=False)

            # TODO: save this mask
            max_len = max(max(len(x) for x in y) for y in data)
            src_mask = generate_square_subsequent_mask(max_len).to(device)

            scores = []
            for batch in data:
                inputs, targets, masks, lengths = build_batch(device, batch)
                seq_len = inputs.shape[0]
                current_mask = src_mask[:seq_len, :seq_len]
                output = self(inputs, current_mask, masks)
                for idx, length in enumerate(lengths):
                    loss = self.criterion(output[:length, idx, :], targets[:length, idx])
                    scores.append(torch.sum(loss))
            scores = utils.unsort(scores, indices)
            scores = list(reversed(scores))
            scores = torch.stack(scores)
            return scores

    def device(self):
        return next(self.parameters()).device

    def save(self, filename):
        params = self.state_dict()
        checkpoint = {
            'config': self.config,
            'params': params,
            'vocab': self.vocab,
        }
        torch.save(checkpoint, filename, _use_new_zipfile_serialization=False)

    @staticmethod
    def load(filename):
        checkpoint = torch.load(filename, lambda storage, loc: storage)
        model = TransformerModel(checkpoint['vocab'],
                                 checkpoint['config'])
        model.load_state_dict(checkpoint['params'])
        return model

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    # TODO: use ours?
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def data_process(vocab, tokenizer, max_len: int, raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    data = filter(lambda t: t.numel() > 0, data)
    if max_len is not None:
        data = filter(lambda t: t.numel() <= max_len, data)
    return utils.sort_with_indices(data, key=len)

def build_batch(device, batch):
    bsz = len(batch)
    # reverse so that the longest is first
    batch = list(reversed(batch))
    max_len = max(len(x) for x in batch) - 1

    # inputs and targets will be 1 smaller than the full sentence
    lengths = [len(x)-1 for x in batch]

    inputs = torch.zeros(max_len, bsz, device=device, dtype=torch.int64)
    targets = torch.zeros(max_len, bsz, device=device, dtype=torch.int64)
    masks = torch.zeros(bsz, max_len, device=device, dtype=torch.bool)
    for line_idx, line in enumerate(batch):
        inputs[:lengths[line_idx], line_idx] = line[:-1]
        targets[:lengths[line_idx], line_idx] = line[1:]
        masks[line_idx, :lengths[line_idx]] = 0
        masks[line_idx, lengths[line_idx]:] = 1
    return (inputs, targets, masks, lengths)

def batchify(data: Tensor, bsz: int, max_len: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    batches = []
    batch_start = 0
    while batch_start < len(data):
        batch_end = batch_start + bsz
        if batch_end > len(data):
            batch_end = len(data)
        batch = data[batch_start:batch_end]
        if max_len is not None and max(len(x) for x in batch) > max_len / 2:
            batch_end = max(batch_start + 1, (batch_end + batch_start) // 2)
            batch = data[batch_start:batch_end]
        batches.append(batch)
        batch_start = batch_end

    return batches

def read_dataset(vocab, tokenizer, filename, batch_size, max_len=None):
    data, _ = data_process(vocab, tokenizer, max_len, open(filename))
    data = batchify(data, batch_size, max_len)
    return data

def train(optimizer, scheduler, epoch, device, train_data, model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 50
    start_time = time.time()
    max_len = max(max(len(x) for x in y) for y in train_data)
    src_mask = generate_square_subsequent_mask(max_len).to(device)

    num_batches = len(train_data)
    random.shuffle(train_data)
    for batch_idx, batch in enumerate(train_data):
        inputs, targets, masks, lengths = build_batch(device, batch)

        # triangle mask for this batch
        seq_len = inputs.shape[0]
        current_mask = src_mask[:seq_len, :seq_len]

        output = model(inputs, current_mask, masks)
        loss = 0
        for idx, length in enumerate(lengths):
            loss = loss + model.criterion(output[:length, idx, :], targets[:length, idx])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % log_interval == 0 and batch_idx > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(device, model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    max_len = max(max(len(x) for x in y) for y in eval_data)
    src_mask = generate_square_subsequent_mask(max_len).to(device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_data):
            inputs, targets, masks, lengths = build_batch(device, batch)

            # triangle mask for this batch
            seq_len = inputs.shape[0]
            current_mask = src_mask[:seq_len, :seq_len]

            output = model(inputs, current_mask, masks)
            for idx, length in enumerate(lengths):
                total_loss += model.criterion(output[:length, idx, :], targets[:length, idx]).item()
    return total_loss / len(eval_data)


DEFAULT_FILES = {
    'vi': {
        'train_file': "vi_vlsp22_train.lm",
        'dev_file': "vi_vlsp22_dev.lm",
        'dev_pred_file': "vi_vlsp22_dev_pred.lm",
        'test_file': "vi_vlsp22_test.lm",
        'test_pred_file': "vi_vlsp22_test_pred.lm",
    },
    'it': {
        'train_file': "it_vit_train_efull.lm",
        'dev_file': "it_vit_dev.lm",
        'dev_pred_file': "it_vit_dev_pred.lm",
        'test_file': "it_vit_test.lm",
        'test_pred_file': "it_vit_test_pred.lm",
    },
}

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int, help='Num epochs to run')
    parser.add_argument('--learning_rate', default=5.0, type=float, help='Initial learning rate')

    parser.add_argument('--data_dir', default="data/trans_lm", help='Where to find the data')
    parser.add_argument('--lang', default='vi', help='Which language to train - sets defaults for the data files, for example')
    parser.add_argument('--train_file', default=None, help='Where to find the data')
    parser.add_argument('--dev_file', default=None, help='Where to find the data')
    parser.add_argument('--dev_pred_file', default=None, help='Where to find the data')
    parser.add_argument('--no_dev_pred', action='store_true', default=False, help="Don't use a val pred file")
    parser.add_argument('--test_file', default=None, help='Where to find the data')
    parser.add_argument('--test_pred_file', default=None, help='Where to find the data')
    parser.add_argument('--no_test_pred', action='store_true', default=False, help="Don't use a test pred file")

    parser.add_argument('--max_len', default=500, type=int, help='Max length of sentence to use')
    parser.add_argument('--no_max_len', action='store_const', const=None, dest='max_len', help='No max_len')
    parser.add_argument('--vocab_min_freq', default=5, type=int, help='Cut off words which appear fewer than # times')

    parser.add_argument('--d_embedding', default=512, type=int, help='Dimension of the embedding at the bottom layer')
    parser.add_argument('--d_hid', default=512, type=int, help='d_hid for the internal layers of the LM')
    parser.add_argument('--dropout', default=0.4, type=float, help='Dropout')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of heads to use')
    parser.add_argument('--n_layers', default=6, type=int, help='Number of layers to use')

    parser.add_argument('--save_dir', type=str, default='saved_models/trans_lm', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    args = parser.parse_args(args=args)

    if args.lang and args.lang in DEFAULT_FILES:
        default_files = DEFAULT_FILES[args.lang]
        for f_type, f_name in default_files.items():
            if getattr(args, f_type) is None:
                setattr(args, f_type, f_name)
            if args.no_test_pred:
                args.test_pred_file = None
            if args.no_dev_pred:
                args.dev_pred_file = None

    return args

def main(args=None):
    random.seed(1234)

    args = parse_args(args)
    utils.log_training_args(args, logger)

    data_dir = args.data_dir
    train_file = os.path.join(data_dir, args.train_file)
    dev_file = os.path.join(data_dir, args.dev_file)
    test_file = os.path.join(data_dir, args.test_file)
    if args.dev_pred_file:
        dev_pred_file = os.path.join(data_dir, args.dev_pred_file)
    if args.test_pred_file:
        test_pred_file = os.path.join(data_dir, args.test_pred_file)

    train_iter = open(train_file)
    tokenizer = lambda x: x.strip().split()
    # (_ROOT is essentially BOS, )_ROOT EOS
    # so perhaps we don't need extra symbols
    # TODO: try to remove dependency on torchtext
    # also, make sure () tokens are not being cut off by vocab_min_freq
    # putting () in the specials will fix that
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), min_freq=args.vocab_min_freq, specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    if not args.save_name:
        args.save_name = args.train_file + ".pt"
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = os.path.join(args.save_dir, args.save_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 20
    eval_batch_size = 10

    train_data = read_dataset(vocab, tokenizer, train_file, batch_size, max_len=500)
    dev_data = read_dataset(vocab, tokenizer, dev_file, eval_batch_size)
    test_data = read_dataset(vocab, tokenizer, test_file, eval_batch_size)
    if args.dev_pred_file:
        dev_pred_data = read_dataset(vocab, tokenizer, dev_pred_file, eval_batch_size)
    if args.test_pred_file:
        test_pred_data = read_dataset(vocab, tokenizer, test_pred_file, eval_batch_size)

    model = TransformerModel(vocab, args).to(device)
    print("Number of tokens in vocab: %s" % len(vocab))

    lr = args.learning_rate  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = args.epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(optimizer, scheduler, epoch, device, train_data, model)
        val_loss = evaluate(device, model, dev_data)
        if args.dev_pred_file:
            val_pred_loss = evaluate(device, model, dev_pred_data)
        #val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        #print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        msg = f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f}'
        if args.dev_pred_file:
            msg += f' | valid pred loss {val_pred_loss:5.2f}'
        print(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            model.save(save_name)
            print("| Saved new best model!")

        print('-' * 89)

        scheduler.step()

    test_loss = evaluate(device, model, test_data)
    if args.test_pred_file:
        test_pred_loss = evaluate(device, model, test_pred_data)
    print('=' * 89)
    #print(f'| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}')
    msg = f'| End of training | test loss {test_loss:5.2f}'
    if args.test_pred_file:
        msg += f' | test pred loss {test_pred_loss:5.2f}'
    print(msg)
    print('=' * 89)

    return best_model

if __name__ == '__main__':
    main()

