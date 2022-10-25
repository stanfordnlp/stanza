import argparse
import math
import os
import random
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import dataset

from torchtext.vocab import build_vocab_from_iterator

import copy
import time

class TransformerModel(nn.Module):

    def __init__(self, vocab, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.vocab = vocab

        self.ntokens = len(vocab)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(self.ntokens, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, self.ntokens)

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
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(output)
        return output

    def save(self, filename):
        params = self.state_dict()
        checkpoint = {
            'params': params,
            'vocab': self.vocab,
        }
        torch.save(checkpoint, filename, _use_new_zipfile_serialization=False)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

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
    data = sorted(data, key=len)
    return data

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

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    batches = []
    for batch_start in range(0, len(data), bsz):
        batch_end = batch_start + bsz
        if batch_end > len(data):
            break
        batch = data[batch_start:batch_end]
        batches.append(batch)

    return batches

def read_dataset(vocab, tokenizer, filename, batch_size, max_len=None):
    data = data_process(vocab, tokenizer, max_len, open(filename))
    data = batchify(data, batch_size)
    return data

def train(criterion, optimizer, scheduler, epoch, device, train_data, model: nn.Module) -> None:
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

        # TODO: handle lengths in masks
        output = model(inputs, current_mask, masks)
        loss = criterion(output.view(-1, model.ntokens), targets.view(-1))

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

def evaluate(criterion, device, model: nn.Module, eval_data: Tensor) -> float:
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

            # TODO: handle lengths in masks
            output = model(inputs, current_mask, masks)
            output_flat = output.view(-1, model.ntokens)
            # TODO: sum the non-masked lengths rather than always multiple by seq_len
            total_loss += seq_len * criterion(output_flat, targets.view(-1)).item()
    return total_loss / (len(eval_data) - 1)

def main():
    random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int, help='Num epochs to run')
    parser.add_argument('--data_dir', default="data/trans_lm", help='Where to find the data')
    parser.add_argument('--train_file', default="it_vit_train.lm", help='Where to find the data')
    parser.add_argument('--max_len', default=500, type=int, help='Max length of sentence to use')
    parser.add_argument('--no_max_len', action='store_const', const=None, dest='max_len', help='No max_len')
    parser.add_argument('--dropout', default=0.4, type=float, help='Dropout')
    parser.add_argument('--nheads', default=4, type=int, help='Number of heads to use')
    parser.add_argument('--nlayers', default=4, type=int, help='Number of layers to use')
    parser.add_argument('--vocab_min_freq', default=5, type=int, help='Cut off words which appear fewer than # times')

    parser.add_argument('--save_dir', type=str, default='saved_models/trans_lm', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    args = parser.parse_args()

    # TODO: make these options
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, args.train_file)
    val_file = os.path.join(data_dir, "it_vit_dev.lm")
    val_pred_file = os.path.join(data_dir, "it_vit_dev_pred.lm")
    test_file = os.path.join(data_dir, "it_vit_test.lm")
    test_pred_file = os.path.join(data_dir, "it_vit_test_pred.lm")

    train_iter = open(train_file)
    tokenizer = lambda x: x.strip().split()
    # (_ROOT is essentially BOS, )_ROOT EOS
    # so perhaps we don't need extra symbols
    # TODO: try to remove dependency on torchtext
    # also, make sure () tokens are not being cut off by vocab_min_freq
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
    val_data = read_dataset(vocab, tokenizer, val_file, eval_batch_size)
    val_pred_data = read_dataset(vocab, tokenizer, val_pred_file, eval_batch_size)
    test_data = read_dataset(vocab, tokenizer, test_file, eval_batch_size)
    test_pred_data = read_dataset(vocab, tokenizer, test_pred_file, eval_batch_size)

    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.nlayers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = args.nheads  # number of heads in nn.MultiheadAttention
    dropout = args.dropout  # dropout probability
    model = TransformerModel(vocab, emsize, nhead, d_hid, nlayers, dropout).to(device)
    print("Number of tokens in vocab: %s" % len(vocab))

    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = args.epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(criterion, optimizer, scheduler, epoch, device, train_data, model)
        val_loss = evaluate(criterion, device, model, val_data)
        val_pred_loss = evaluate(criterion, device, model, val_pred_data)
        #val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        #print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | valid pred loss {val_pred_loss:5.2f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            model.save(save_name)
            print("| Saved new best model!")

        print('-' * 89)

        scheduler.step()

    test_loss = evaluate(criterion, device, model, test_data)
    test_pred_loss = evaluate(criterion, device, model, test_pred_data)
    print('=' * 89)
    #print(f'| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}')
    print(f'| End of training | test loss {test_loss:5.2f} | test pred loss {test_pred_loss:5.2f}')
    print('=' * 89)

if __name__ == '__main__':
    main()

