"""
Entry point for training and evaluating a character-level neural language model.
"""

import random
import argparse
from copy import copy
import numpy as np
import torch
import math
import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG)

from stanfordnlp.models.common.char_model import CharacterLanguageModel
from stanfordnlp.models.pos.vocab import CharVocab, CommonCharVocab
from stanfordnlp.models.common import utils

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

def load_data(path, vocab, direction):
    lines = open(path).readlines() # reserve '\n'
    data = list(''.join(lines))
    idx = vocab['char'].map(data)
    if direction == 'backward': idx = idx[::-1]
    return torch.tensor(idx)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help="Input plaintext file")
    parser.add_argument('--eval_file', type=str, help="Input plaintext file for the dev/test set")
    parser.add_argument('--lang', type=str, help="Language")
    parser.add_argument('--shorthand', type=str, help="UD treebank shorthand")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--direction', default='forward', choices=['forward', 'backward'], help="Forward or backward language model")

    parser.add_argument('--char_emb_dim', type=int, default=100, help="Dimension of unit embeddings")
    parser.add_argument('--char_hidden_dim', type=int, default=2048, help="Dimension of hidden units")
    parser.add_argument('--char_num_layers', type=int, default=1, help="Layers of RNN in the language model")
    parser.add_argument('--char_dropout', type=float, default=0.05, help="Dropout probability")
    parser.add_argument('--char_rec_dropout', type=float, default=0.0, help="Recurrent dropout probability")

    parser.add_argument('--batch_size', type=int, default=100, help="Batch size to use")
    parser.add_argument('--bptt_size', type=int, default=70, help="Sequence length to consider at a time") # TODO: determine
    parser.add_argument('--epochs', type=int, default=20, help="Total epochs to train the model for")
    parser.add_argument('--max_grad_norm', type=float, default=0.25, help="Maximum gradient norm to clip to")
    parser.add_argument('--lr0', type=float, default=20, help="Initial learning rate")

    parser.add_argument('--anneal', type=float, default=.999, help="Anneal the learning rate by this amount when dev performance deteriorate")
    parser.add_argument('--anneal_after', type=int, default=2000, help="Anneal the learning rate no earlier than this step")
    

    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--steps', type=int, default=20000, help="Steps to train the model for, if unspecified use epochs")
    parser.add_argument('--report_steps', type=int, default=20, help="Update step interval to report loss")
    parser.add_argument('--shuffle_steps', type=int, default=100, help="Step interval to shuffle each paragragraph in the generator")
    parser.add_argument('--eval_steps', type=int, default=200, help="Step interval to evaluate the model on the dev set for early stopping")
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--load_name', type=str, default=None, help="File name to load a saved model")
    parser.add_argument('--save_dir', type=str, default='saved_models/charlm', help="Directory to save models in")
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
    print("Running character-level language model in {} mode".format(args['mode']))
    
    utils.ensure_dir(args['save_dir'])

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def train_epoch(args, vocab, batches, model, params, optimizer, criterion):
    # TODO: shuffle the data
    # TODO: decrease lr
    model.train()
    hidden = None
    total_loss = 0.0

    for iteration, i in enumerate(range(0, batches.size(1), args['bptt_size'])):
        data, target = get_batch(batches, i, args['bptt_size'])
        lens = [data.size(1) for i in range(data.size(0))]
        if args['cuda']: 
            data.cuda()
            target.cuda()        
        
        optimizer.zero_grad()

        output, hidden, decoded = model.forward(data, lens, hidden)

        loss = criterion(decoded.view(-1, len(vocab['char'])), target)
        total_loss += loss.data.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(params, args['max_grad_norm'])
        optimizer.step()

        hidden = repackage_hidden(hidden)

        if iteration % args['report_steps'] == 0 and iteration > 0:
            cur_loss = total_loss / args['report_steps']
            print(
                "| {:5d}/{:5d} batches | loss {:5.2f} | ppl {:8.2f}".format(
                    iteration,
                    batches.size(1) // args['bptt_size'],
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0.0
    return

def evaluate_epoch(args, vocab, batches, model, criterion):
    model.eval()
    hidden = None
    total_loss = 0

    with torch.no_grad():
        for i in range(0, batches.size(1), args['bptt_size']):
            data, target = get_batch(batches, i, args['bptt_size'])
            lens = [data.size(1) for i in range(data.size(0))]
            if args['cuda']: 
                data.cuda()
                target.cuda()     

            output, hidden, decoded = model.forward(data, lens, hidden)
            loss = criterion(decoded.view(-1, len(vocab['char'])), target)
            
            hidden = repackage_hidden(hidden)
            total_loss += data.size(1) * loss.data.item()
    return total_loss / batches.size(1)
           
def train(args):
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_charlm.pt'.format(args['save_dir'], args['shorthand'])

    vocab = {'char': CommonCharVocab([])}

    train_data = load_data(args['train_file'], vocab, args['direction'])
    train_batches = batchify(train_data, args['batch_size'])

    dev_data = load_data(args['eval_file'], vocab, args['direction'])
    dev_batches = batchify(dev_data, args['batch_size'])

    model = CharacterLanguageModel(args, vocab, is_forward_lm=True if args['direction'] == 'forward' else False)
    if args['cuda']: model.cuda()
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args['lr0'])
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = None
    for epoch in range(args['epochs']):
        train_epoch(args, vocab, train_batches, model, params, optimizer, criterion)
        loss = evaluate_epoch(args, vocab, dev_batches, model, criterion)
        print(
            "| {:5d}/{:5d} epochs | loss {:5.2f} | ppl {:8.2f}".format(
                epoch,
                args['epochs'],
                loss,
                math.exp(loss),
            )
        )
        if best_loss is None or loss < best_loss:
            best_loss = loss
            model.save(model_file)
            print('new best model saved.')
    

def evaluate(args):
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_charlm.pt'.format(args['save_dir'], args['shorthand'])

    model = CharacterLanguageModel.load(model_file)
    vocab = model.vocab
    data = load_data(args['eval_file'], vocab, args['direction'])
    batches = batchify(data, args['batch_size'])
    criterion = torch.nn.CrossEntropyLoss()

    loss = evaluate_epoch(args, vocab, batches, model, criterion)
    print(
        "| best model | loss {:5.2f} | ppl {:8.2f}".format(
            loss,
            math.exp(loss),
        )
    )

if __name__ == '__main__':
    main()
