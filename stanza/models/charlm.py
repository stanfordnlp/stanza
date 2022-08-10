"""
Entry point for training and evaluating a character-level neural language model.
"""

import argparse
from copy import copy
import logging
import lzma
import math
import os
import random
import time
from types import GeneratorType
import numpy as np
import torch

from stanza.models.common.char_model import build_charlm_vocab, CharacterLanguageModel, CharacterLanguageModelTrainer
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

def load_file(filename, vocab, direction):
    with utils.open_read_text(filename) as fin:
        data = fin.read()

    idx = vocab['char'].map(data)
    if direction == 'backward': idx = idx[::-1]
    return torch.tensor(idx)

def load_data(path, vocab, direction):
    if os.path.isdir(path):
        filenames = sorted(os.listdir(path))
        for filename in filenames:
            logger.info('Loading data from {}'.format(filename))
            data = load_file(os.path.join(path, filename), vocab, direction)
            yield data
    else:
        data = load_file(path, vocab, direction)
        yield data

def parse_args(args=None):
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
    parser.add_argument('--lr0', type=float, default=5, help="Initial learning rate")
    parser.add_argument('--anneal', type=float, default=0.25, help="Anneal the learning rate by this amount when dev performance deteriorate")
    parser.add_argument('--patience', type=int, default=1, help="Patience for annealing the learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for SGD.')
    parser.add_argument('--cutoff', type=int, default=1000, help="Frequency cutoff for char vocab. By default we assume a very large corpus.")
    
    parser.add_argument('--report_steps', type=int, default=50, help="Update step interval to report loss")
    parser.add_argument('--eval_steps', type=int, default=100000, help="Update step interval to run eval on dev; set to -1 to eval after each epoch")
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--vocab_save_name', type=str, default=None, help="File name to save the vocab")
    parser.add_argument('--checkpoint_save_name', type=str, default=None, help="File name to save the most recent checkpoint")
    parser.add_argument('--no_checkpoint', dest='checkpoint', action='store_false', help="Don't save checkpoints")
    parser.add_argument('--save_dir', type=str, default='saved_models/charlm', help="Directory to save models in")
    parser.add_argument('--summary', action='store_true', help='Use summary writer to record progress.')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA and run on CPU.')
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--wandb', action='store_true', help='Start a wandb session and write the results of training.  Only applies to training.  Use --wandb_name instead to specify a name')
    parser.add_argument('--wandb_name', default=None, help='Name of a wandb session to start when training.  Will default to the dataset short name')

    args = parser.parse_args(args=args)

    if args.wandb_name:
        args.wandb = True

    args = vars(args)
    return args

def main(args=None):
    args = parse_args(args=args)

    if args['cpu']:
        args['cuda'] = False
    utils.set_random_seed(args['seed'], args['cuda'])

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

def evaluate_and_save(args, vocab, data, trainer, best_loss, model_file, checkpoint_file, writer=None):
    """
    Run an evaluation over entire dataset, print progress and save the model if necessary.
    """
    start_time = time.time()
    loss = evaluate_epoch(args, vocab, data, trainer.model, trainer.criterion)
    ppl = math.exp(loss)
    elapsed = int(time.time() - start_time)
    # TODO: step the scheduler less often when the eval frequency is higher
    trainer.scheduler.step(loss)
    logger.info(
        "| eval checkpoint @ global step {:10d} | time elapsed {:6d}s | loss {:5.2f} | ppl {:8.2f}".format(
            trainer.global_step,
            elapsed,
            loss,
            ppl,
        )
    )
    if best_loss is None or loss < best_loss:
        best_loss = loss
        trainer.save(model_file, full=False)
        logger.info('new best model saved at step {:10d}'.format(trainer.global_step))
    if writer:
        writer.add_scalar('dev_loss', loss, global_step=trainer.global_step)
        writer.add_scalar('dev_ppl', ppl, global_step=trainer.global_step)
    if checkpoint_file:
        trainer.save(checkpoint_file, full=True)
        logger.info('new checkpoint saved at step {:10d}'.format(trainer.global_step))

    return loss, ppl, best_loss

def get_current_lr(trainer, args):
    return trainer.scheduler.state_dict().get('_last_lr', [args['lr0']])[0]

def load_char_vocab(vocab_file):
    return {'char': CharVocab.load_state_dict(torch.load(vocab_file, lambda storage, loc: storage))}

def train(args):
    if args['save_name']:
        save_name = args['save_name']
    else:
        save_name = '{}_{}_charlm.pt'.format(args['shorthand'], args['direction'])
    model_file = os.path.join(args['save_dir'], save_name)

    vocab_file = args['save_dir'] + '/' + args['vocab_save_name'] if args['vocab_save_name'] is not None \
        else '{}/{}_vocab.pt'.format(args['save_dir'], args['shorthand'])

    if args['checkpoint']:
        checkpoint_file = utils.checkpoint_name(args['save_dir'], save_name, args['checkpoint_save_name'])
    else:
        checkpoint_file = None

    if os.path.exists(vocab_file):
        logger.info('Loading existing vocab file')
        vocab = load_char_vocab(vocab_file)
    else:
        logger.info('Building and saving vocab')
        vocab = {'char': build_charlm_vocab(args['train_file'] if args['train_dir'] is None else args['train_dir'], cutoff=args['cutoff'])}
        torch.save(vocab['char'].state_dict(), vocab_file)
    logger.info("Training model with vocab size: {}".format(len(vocab['char'])))

    if checkpoint_file and os.path.exists(checkpoint_file):
        logger.info('Loading existing checkpoint: %s' % checkpoint_file)
        trainer = CharacterLanguageModelTrainer.load(args, checkpoint_file, finetune=True)
    else:
        trainer = CharacterLanguageModelTrainer.from_new_model(args, vocab)

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

    if args['wandb']:
        import wandb
        wandb_name = args['wandb_name'] if args['wandb_name'] else '%s_%s_charlm' % (args['shorthand'], args['direction'])
        wandb.init(name=wandb_name, config=args)
        wandb.run.define_metric('best_loss', summary='min')
        wandb.run.define_metric('ppl', summary='min')

    best_loss = None
    start_epoch = trainer.epoch  # will default to 1 for a new trainer
    for trainer.epoch in range(start_epoch, args['epochs']+1):
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
                trainer.model.train()
                trainer.global_step += 1
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
                
                trainer.optimizer.zero_grad()
                output, hidden, decoded = trainer.model.forward(data, lens, hidden)
                loss = trainer.criterion(decoded.view(-1, len(vocab['char'])), target)
                total_loss += loss.data.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(trainer.params, args['max_grad_norm'])
                trainer.optimizer.step()

                hidden = repackage_hidden(hidden)

                if (iteration + 1) % args['report_steps'] == 0:
                    cur_loss = total_loss / args['report_steps']
                    elapsed = time.time() - start_time
                    logger.info(
                        "| epoch {:5d} | {:5d}/{:5d} batches | sec/batch {:.6f} | loss {:5.2f} | ppl {:8.2f}".format(
                            trainer.epoch,
                            iteration + 1,
                            total_batches,
                            elapsed / args['report_steps'],
                            cur_loss,
                            math.exp(cur_loss),
                        )
                    )
                    if args['wandb']:
                        wandb.log({'train_loss': cur_loss}, step=trainer.global_step)
                    total_loss = 0.0

                iteration += 1
                i += seq_len

                # evaluate if necessary
                if eval_within_epoch and trainer.global_step % args['eval_steps'] == 0:
                    _, ppl, best_loss = evaluate_and_save(args, vocab, dev_data, trainer, best_loss, model_file, checkpoint_file, writer)
                    if args['wandb']:
                        wandb.log({'ppl': ppl, 'best_loss': best_loss, 'lr': get_current_lr(trainer, args)}, step=trainer.global_step)

        # if eval_interval isn't provided, run evaluation after each epoch
        if not eval_within_epoch or trainer.epoch == args['epochs']:
            _, ppl, best_loss = evaluate_and_save(args, vocab, dev_data, trainer, best_loss, model_file, checkpoint_file, writer)
            if args['wandb']:
                wandb.log({'ppl': ppl, 'best_loss': best_loss, 'lr': get_current_lr(trainer, args)}, step=trainer.global_step)

    if writer:
        writer.close()
    if args['wandb']:
        wandb.finish()
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
