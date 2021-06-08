"""
Entry point for training and evaluating a character-level neural language model.
"""

import random
import argparse
from copy import copy
from collections import Counter
from types import GeneratorType
import numpy as np
import torch
import math
import logging
import time
import os
import lzma

from stanza.models.common.char_model import CharacterLanguageModel
from stanza.models.common.vocab import CharVocab
from stanza.models.common import utils
from stanza.models import _training_logging

logger = logging.getLogger('stanza')

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1) # batch_first is True
    return data

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, source.size(1) - 1 - i)
    data = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len].reshape(-1)
    return data, target

def readlines(path):
    if path.endswith(".xz"):
        with lzma.open(path, mode='rt') as fin:
            lines = fin.readlines()
    else:
        with open(path) as fin:
            lines = fin.readlines()  # preserve '\n'
    return lines

def build_vocab(path, cutoff=0):
    # Requires a large amount of memory, but only need to build once

    # here we need some trick to deal with excessively large files
    # for each file we accumulate the counter of characters, and
    # at the end we simply pass a list of chars to the vocab builder
    counter = Counter()
    if os.path.isdir(path):
        filenames = sorted(os.listdir(path))
    else:
        filenames = [path]
    for filename in filenames:
        lines = readlines(path + '/' + filename)
        for line in lines:
            counter.update(list(line))
    # remove infrequent characters from vocab
    for k in list(counter.keys()):
        if counter[k] < cutoff:
            del counter[k]
    # a singleton list of all characters
    data = [sorted([x[0] for x in counter.most_common()])]
    vocab = CharVocab(data) # skip cutoff argument because this has been dealt with
    return vocab

def load_file(path, vocab, direction):
    lines = readlines(path)
    data = list(''.join(lines))
    idx = vocab['char'].map(data)
    if direction == 'backward': idx = idx[::-1]
    return torch.tensor(idx)

def load_data(path, vocab, direction):
    if os.path.isdir(path):
        filenames = sorted(os.listdir(path))
        for filename in filenames:
            logger.info('Loading data from {}'.format(filename))
            data = load_file(path + '/' + filename, vocab, direction)
            yield data
    else:
        data = load_file(path, vocab, direction)
        yield data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help="Input plaintext file")
    parser.add_argument('--train_dir', type=str, help="If non-empty, load from directory with multiple training files")
    parser.add_argument('--eval_file', type=str, help="Input plaintext file for the dev/test set")
    parser.add_argument('--lang', type=str, help="Language")
    parser.add_argument('--shorthand', type=str, help="UD treebank shorthand")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--direction', default='forward', choices=['forward', 'backward'], help="Forward or backward language model")

    parser.add_argument('--char_emb_dim', type=int, default=100, help="Dimension of unit embeddings")
    parser.add_argument('--char_hidden_dim', type=int, default=1024, help="Dimension of hidden units")
    parser.add_argument('--char_num_layers', type=int, default=1, help="Layers of RNN in the language model")
    parser.add_argument('--char_dropout', type=float, default=0.05, help="Dropout probability")
    parser.add_argument('--char_unit_dropout', type=float, default=1e-5, help="Randomly set an input char to UNK during training")
    parser.add_argument('--char_rec_dropout', type=float, default=0.0, help="Recurrent dropout probability")

    parser.add_argument('--batch_size', type=int, default=100, help="Batch size to use")
    parser.add_argument('--bptt_size', type=int, default=250, help="Sequence length to consider at a time")
    parser.add_argument('--epochs', type=int, default=50, help="Total epochs to train the model for")
    parser.add_argument('--max_grad_norm', type=float, default=0.25, help="Maximum gradient norm to clip to")
    parser.add_argument('--lr0', type=float, default=20, help="Initial learning rate")
    parser.add_argument('--anneal', type=float, default=0.25, help="Anneal the learning rate by this amount when dev performance deteriorate")
    parser.add_argument('--patience', type=int, default=1, help="Patience for annealing the learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for SGD.')
    parser.add_argument('--cutoff', type=int, default=1000, help="Frequency cutoff for char vocab. By default we assume a very large corpus.")
    
    parser.add_argument('--report_steps', type=int, default=50, help="Update step interval to report loss")
    parser.add_argument('--eval_steps', type=int, default=-1, help="Update step interval to run eval on dev; set to -1 to eval after each epoch")
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--vocab_save_name', type=str, default=None, help="File name to save the vocab")
    parser.add_argument('--save_dir', type=str, default='saved_models/charlm', help="Directory to save models in")
    parser.add_argument('--summary', action='store_true', help='Use summary writer to record progress.')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA and run on CPU.')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = vars(args)
    logger.info("Running {} character-level language model in {} mode".format(args['direction'], args['mode']))
    
    utils.ensure_dir(args['save_dir'])

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def evaluate_epoch(args, vocab, data, model, criterion):
    """
    Run an evaluation over entire dataset.
    """
    model.eval()
    hidden = None
    total_loss = 0
    if isinstance(data, GeneratorType):
        data = list(data)
        assert len(data) == 1, 'Only support single dev/test file'
        data = data[0]
    batches = batchify(data, args['batch_size'])
    with torch.no_grad():
        for i in range(0, batches.size(1) - 1, args['bptt_size']):
            data, target = get_batch(batches, i, args['bptt_size'])
            lens = [data.size(1) for i in range(data.size(0))]
            if args['cuda']:
                data = data.cuda()
                target = target.cuda()

            output, hidden, decoded = model.forward(data, lens, hidden)
            loss = criterion(decoded.view(-1, len(vocab['char'])), target)
            
            hidden = repackage_hidden(hidden)
            total_loss += data.size(1) * loss.data.item()
    return total_loss / batches.size(1)

def evaluate_and_save(args, vocab, data, model, criterion, scheduler, best_loss, global_step, model_file, writer=None):
    """
    Run an evaluation over entire dataset, print progress and save the model if necessary.
    """
    start_time = time.time()
    loss = evaluate_epoch(args, vocab, data, model, criterion)
    ppl = math.exp(loss)
    elapsed = int(time.time() - start_time)
    scheduler.step(loss)
    logger.info(
        "| eval checkpoint @ global step {:10d} | time elapsed {:6d}s | loss {:5.2f} | ppl {:8.2f}".format(
            global_step,
            elapsed,
            loss,
            ppl,
        )
    )
    if best_loss is None or loss < best_loss:
        best_loss = loss
        model.save(model_file)
        logger.info('new best model saved at step {:10d}.'.format(global_step))
    if writer:
        writer.add_scalar('dev_loss', loss, global_step=global_step)
        writer.add_scalar('dev_ppl', ppl, global_step=global_step)
    return loss, ppl, best_loss

def train(args):
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_{}_charlm.pt'.format(args['save_dir'], args['shorthand'], args['direction'])
    vocab_file = args['save_dir'] + '/' + args['vocab_save_name'] if args['vocab_save_name'] is not None \
        else '{}/{}_vocab.pt'.format(args['save_dir'], args['shorthand'])

    if os.path.exists(vocab_file):
        logger.info('Loading existing vocab file')
        vocab = {'char': CharVocab.load_state_dict(torch.load(vocab_file, lambda storage, loc: storage))}
    else:
        logger.info('Building and saving vocab')
        vocab = {'char': build_vocab(args['train_file'] if args['train_dir'] is None else args['train_dir'], cutoff=args['cutoff'])}
        torch.save(vocab['char'].state_dict(), vocab_file)
    logger.info("Training model with vocab size: {}".format(len(vocab['char'])))

    model = CharacterLanguageModel(args, vocab, is_forward_lm=True if args['direction'] == 'forward' else False)
    if args['cuda']: model = model.cuda()
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args['lr0'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=args['anneal'], patience=args['patience'])

    writer = None
    if args['summary']:
        from torch.utils.tensorboard import SummaryWriter
        summary_dir = '{}/{}_summary'.format(args['save_dir'], args['save_name']) if args['save_name'] is not None \
            else '{}/{}_{}_charlm_summary'.format(args['save_dir'], args['shorthand'], args['direction'])
        writer = SummaryWriter(log_dir=summary_dir)
    
    # evaluate model within epoch if eval_interval is set
    eval_within_epoch = False
    if args['eval_steps'] > 0:
        eval_within_epoch = True

    best_loss = None
    global_step = 0
    for epoch in range(1,args['epochs']+1):
        # load train data from train_dir if not empty, otherwise load from file
        if args['train_dir'] is not None:
            train_path = args['train_dir']
        else:
            train_path = args['train_file']
        train_data = load_data(train_path, vocab, args['direction'])
        dev_data = load_file(args['eval_file'], vocab, args['direction']) # dev must be a single file

        # run over entire training set
        for data_chunk in train_data:
            batches = batchify(data_chunk, args['batch_size'])
            hidden = None
            total_loss = 0.0
            total_batches = math.ceil((batches.size(1) - 1) / args['bptt_size'])
            iteration, i = 0, 0
            # over the data chunk
            while i < batches.size(1) - 1 - 1:
                model.train()
                global_step += 1
                start_time = time.time()
                bptt = args['bptt_size'] if np.random.random() < 0.95 else args['bptt_size']/ 2.
                # prevent excessively small or negative sequence lengths
                seq_len = max(5, int(np.random.normal(bptt, 5)))
                # prevent very large sequence length, must be <= 1.2 x bptt
                seq_len = min(seq_len, int(args['bptt_size'] * 1.2))
                data, target = get_batch(batches, i, seq_len)
                lens = [data.size(1) for i in range(data.size(0))]
                if args['cuda']:
                    data = data.cuda()
                    target = target.cuda()
                
                optimizer.zero_grad()
                output, hidden, decoded = model.forward(data, lens, hidden)
                loss = criterion(decoded.view(-1, len(vocab['char'])), target)
                total_loss += loss.data.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, args['max_grad_norm'])
                optimizer.step()

                hidden = repackage_hidden(hidden)

                if (iteration + 1) % args['report_steps'] == 0:
                    cur_loss = total_loss / args['report_steps']
                    elapsed = time.time() - start_time
                    logger.info(
                        "| epoch {:5d} | {:5d}/{:5d} batches | sec/batch {:.6f} | loss {:5.2f} | ppl {:8.2f}".format(
                            epoch,
                            iteration + 1,
                            total_batches,
                            elapsed / args['report_steps'],
                            cur_loss,
                            math.exp(cur_loss),
                        )
                    )
                    total_loss = 0.0

                iteration += 1
                i += seq_len

                # evaluate if necessary
                if eval_within_epoch and global_step % args['eval_steps'] == 0:
                    _, _, best_loss = evaluate_and_save(args, vocab, dev_data, model, criterion, scheduler, best_loss, \
                        global_step, model_file, writer)

        # if eval_interval isn't provided, run evaluation after each epoch
        if not eval_within_epoch:
            _, _, best_loss = evaluate_and_save(args, vocab, dev_data, model, criterion, scheduler, best_loss, \
                        epoch, model_file, writer) # use epoch in place of global_step for logging

    if writer:
        writer.close()
    return

def evaluate(args):
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_{}_charlm.pt'.format(args['save_dir'], args['shorthand'], args['direction'])

    model = CharacterLanguageModel.load(model_file)
    if args['cuda']: model = model.cuda()
    vocab = model.vocab
    data = load_data(args['eval_file'], vocab, args['direction'])
    criterion = torch.nn.CrossEntropyLoss()
    
    loss = evaluate_epoch(args, vocab, data, model, criterion)
    logger.info(
        "| best model | loss {:5.2f} | ppl {:8.2f}".format(
            loss,
            math.exp(loss),
        )
    )
    return

if __name__ == '__main__':
    main()
