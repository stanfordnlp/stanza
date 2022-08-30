"""
Entry point for training and evaluating a neural tokenizer.

This tokenizer treats tokenization and sentence segmentation as a tagging problem, and uses a combination of
recurrent and convolutional architectures.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.

Updated: This new version of tokenizer model incorporates the dictionary feature, especially useful for languages that
have multi-syllable words such as Vietnamese, Chinese or Thai. In summary, a lexicon contains all unique words found in 
training dataset and external lexicon (if any) is created during training and saved alongside the model after training.
Using this lexicon, a dictionary is created which includes "words", "prefixes" and "suffixes" sets. During data preparation,
dictionary features are extracted at each character position, to "look ahead" and "look backward" to see if any words formed
found in the dictionary. The window size (or the dictionary feature length) is defined at the 95-percentile among all the existing
words in the lexicon, this is to eliminate the less frequent but long words (avoid having a high-dimension feat vector). Prefixes 
and suffixes are used to stop early during the window-dictionary checking process.  
"""

import argparse
from copy import copy
import logging
import random
import numpy as np
import os
import torch
import json
from stanza.models.common import utils
from stanza.models.tokenization.trainer import Trainer
from stanza.models.tokenization.data import DataLoader, TokenizationDataset
from stanza.models.tokenization.utils import load_mwt_dict, eval_model, output_predictions, load_lexicon, create_dictionary
from stanza.models import _training_logging

logger = logging.getLogger('stanza')

def parse_args(args=None):
    """
    If args == None, the system args are used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--mwt_json_file', type=str, default=None, help="JSON file for MWT expansions")
    parser.add_argument('--conll_file', type=str, default=None, help="CoNLL file for output")
    parser.add_argument('--dev_txt_file', type=str, help="(Train only) Input plaintext file for the dev set")
    parser.add_argument('--dev_label_file', type=str, default=None, help="(Train only) Character-level label file for the dev set")
    parser.add_argument('--dev_conll_gold', type=str, default=None, help="(Train only) CoNLL-U file for the dev set for early stopping")
    parser.add_argument('--lang', type=str, help="Language")
    parser.add_argument('--shorthand', type=str, help="UD treebank shorthand")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--skip_newline', action='store_true', help="Whether to skip newline characters in input. Particularly useful for languages like Chinese.")

    parser.add_argument('--emb_dim', type=int, default=32, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Dimension of hidden units")
    parser.add_argument('--conv_filters', type=str, default="1,9", help="Configuration of conv filters. ,, separates layers and , separates filter sizes in the same layer.")
    parser.add_argument('--no-residual', dest='residual', action='store_false', help="Add linear residual connections")
    parser.add_argument('--no-hierarchical', dest='hierarchical', action='store_false', help="\"Hierarchical\" RNN tokenizer")
    parser.add_argument('--hier_invtemp', type=float, default=0.5, help="Inverse temperature used in propagating tokenization predictions between RNN layers")
    parser.add_argument('--input_dropout', action='store_true', help="Dropout input embeddings as well")
    parser.add_argument('--conv_res', type=str, default=None, help="Convolutional residual layers for the RNN")
    parser.add_argument('--rnn_layers', type=int, default=1, help="Layers of RNN in the tokenizer")
    parser.add_argument('--use_dictionary', action='store_true', help="Use dictionary feature. The lexicon is created using the training data and external dict (if any) expected to be found under the same folder of training dataset, formatted as SHORTHAND-externaldict.txt where each line in this file is a word. For example, data/tokenize/zh_gsdsimp-externaldict.txt")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm to clip to")
    parser.add_argument('--anneal', type=float, default=.999, help="Anneal the learning rate by this amount when dev performance deteriorate")
    parser.add_argument('--anneal_after', type=int, default=2000, help="Anneal the learning rate no earlier than this step")
    parser.add_argument('--lr0', type=float, default=2e-3, help="Initial learning rate")
    parser.add_argument('--dropout', type=float, default=0.33, help="Dropout probability")
    parser.add_argument('--unit_dropout', type=float, default=0.33, help="Unit dropout probability")
    parser.add_argument('--feat_dropout', type=float, default=0.05, help="Features dropout probability for each element in feature vector")
    parser.add_argument('--feat_unit_dropout', type=float, default=0.33, help="The whole feature of units dropout probability")
    parser.add_argument('--tok_noise', type=float, default=0.02, help="Probability to induce noise to the input of the higher RNN")
    parser.add_argument('--sent_drop_prob', type=float, default=0.2, help="Probability to drop sentences at the end of batches during training uniformly at random.  Idea is to fake paragraph endings.")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--max_seqlen', type=int, default=100, help="Maximum sequence length to consider at a time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use")
    parser.add_argument('--epochs', type=int, default=10, help="Total epochs to train the model for")
    parser.add_argument('--steps', type=int, default=50000, help="Steps to train the model for, if unspecified use epochs")
    parser.add_argument('--report_steps', type=int, default=20, help="Update step interval to report loss")
    parser.add_argument('--shuffle_steps', type=int, default=100, help="Step interval to shuffle each paragraph in the generator")
    parser.add_argument('--eval_steps', type=int, default=200, help="Step interval to evaluate the model on the dev set for early stopping")
    parser.add_argument('--max_steps_before_stop', type=int, default=5000, help='Early terminates after this many steps if the dev scores are not improving')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--load_name', type=str, default=None, help="File name to load a saved model")
    parser.add_argument('--save_dir', type=str, default='saved_models/tokenize', help="Directory to save models in")
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA and run on CPU.')
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--use_mwt', dest='use_mwt', default=None, action='store_true', help='Whether or not to include mwt output layers.  If set to None, this will be determined by examining the training data for MWTs')
    parser.add_argument('--no_use_mwt', dest='use_mwt', action='store_false', help='Whether or not to include mwt output layers')

    parser.add_argument('--wandb', action='store_true', help='Start a wandb session and write the results of training.  Only applies to training.  Use --wandb_name instead to specify a name')
    parser.add_argument('--wandb_name', default=None, help='Name of a wandb session to start when training.  Will default to the dataset short name')

    args = parser.parse_args(args=args)

    if args.wandb_name:
        args.wandb = True

    return args

def main(args=None):
    args = parse_args(args=args)

    if args.cpu:
        args.cuda = False
    utils.set_random_seed(args.seed, args.cuda)

    args = vars(args)
    logger.info("Running tokenizer in {} mode".format(args['mode']))

    args['feat_funcs'] = ['space_before', 'capitalized', 'numeric', 'end_of_para', 'start_of_para']
    args['feat_dim'] = len(args['feat_funcs'])
    save_name = args['save_name'] if args['save_name'] else '{}_tokenizer.pt'.format(args['shorthand'])
    args['save_name'] = os.path.join(args['save_dir'], save_name)
    utils.ensure_dir(args['save_dir'])

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def train(args):
    if args['use_dictionary']:
        #load lexicon
        lexicon, args['num_dict_feat'] = load_lexicon(args)
        #create the dictionary
        dictionary = create_dictionary(lexicon)
        #adjust the feat_dim
        args['feat_dim'] += args['num_dict_feat']*2
    else:
        args['num_dict_feat'] = 0
        lexicon=None
        dictionary=None

    mwt_dict = load_mwt_dict(args['mwt_json_file'])

    train_input_files = {
            'txt': args['txt_file'],
            'label': args['label_file']
            }
    train_batches = DataLoader(args, input_files=train_input_files, dictionary=dictionary)
    vocab = train_batches.vocab

    args['vocab_size'] = len(vocab)

    dev_input_files = {
            'txt': args['dev_txt_file'],
            'label': args['dev_label_file']
            }
    dev_batches = TokenizationDataset(args, input_files=dev_input_files, vocab=vocab, evaluation=True, dictionary=dictionary)

    if args['use_mwt'] is None:
        args['use_mwt'] = train_batches.has_mwt()
        logger.info("Found {}mwts in the training data.  Setting use_mwt to {}".format(("" if args['use_mwt'] else "no "), args['use_mwt']))

    trainer = Trainer(args=args, vocab=vocab, lexicon=lexicon, dictionary=dictionary, use_cuda=args['cuda'])

    if args['load_name'] is not None:
        load_name = os.path.join(args['save_dir'], args['load_name'])
        trainer.load(load_name)
    trainer.change_lr(args['lr0'])

    N = len(train_batches)
    steps = args['steps'] if args['steps'] is not None else int(N * args['epochs'] / args['batch_size'] + .5)
    lr = args['lr0']

    prev_dev_score = -1
    best_dev_score = -1
    best_dev_step = -1

    if args['wandb']:
        import wandb
        wandb_name = args['wandb_name'] if args['wandb_name'] else "%s_tokenizer" % args['shorthand']
        wandb.init(name=wandb_name, config=args)
        wandb.run.define_metric('train_loss', summary='min')
        wandb.run.define_metric('dev_score', summary='max')


    for step in range(1, steps+1):
        batch = train_batches.next(unit_dropout=args['unit_dropout'], feat_unit_dropout = args['feat_unit_dropout'])

        loss = trainer.update(batch)
        if step % args['report_steps'] == 0:
            logger.info("Step {:6d}/{:6d} Loss: {:.3f}".format(step, steps, loss))
            if args['wandb']:
                wandb.log({'train_loss': loss}, step=step)

        if args['shuffle_steps'] > 0 and step % args['shuffle_steps'] == 0:
            train_batches.shuffle()

        if step % args['eval_steps'] == 0:
            dev_score = eval_model(args, trainer, dev_batches, vocab, mwt_dict)
            if args['wandb']:
                wandb.log({'dev_score': dev_score}, step=step)
            reports = ['Dev score: {:6.3f}'.format(dev_score * 100)]
            if step >= args['anneal_after'] and dev_score < prev_dev_score:
                reports += ['lr: {:.6f} -> {:.6f}'.format(lr, lr * args['anneal'])]
                lr *= args['anneal']
                trainer.change_lr(lr)

            prev_dev_score = dev_score

            if dev_score > best_dev_score:
                reports += ['New best dev score!']
                best_dev_score = dev_score
                best_dev_step = step
                trainer.save(args['save_name'])
            elif best_dev_step > 0 and step - best_dev_step > args['max_steps_before_stop']:
                reports += ['Stopping training after {} steps with no improvement'.format(step - best_dev_step)]
                logger.info('\t'.join(reports))
                break

            logger.info('\t'.join(reports))

    if args['wandb']:
        wandb.finish()

    if best_dev_step > -1:
        logger.info('Best dev score={} at step {}'.format(best_dev_score, best_dev_step))
    else:
        logger.info('Dev set never evaluated.  Saving final model')
        trainer.save(args['save_name'])

def evaluate(args):
    mwt_dict = load_mwt_dict(args['mwt_json_file'])
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=args['load_name'] or args['save_name'], use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    for k in loaded_args:
        if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'load_name', 'save_name']:
            args[k] = loaded_args[k]
    
    eval_input_files = {
            'txt': args['txt_file'],
            'label': args['label_file']
            }


    batches = TokenizationDataset(args, input_files=eval_input_files, vocab=vocab, evaluation=True, dictionary=trainer.dictionary)

    oov_count, N, _, _ = output_predictions(args['conll_file'], trainer, batches, vocab, mwt_dict, args['max_seqlen'])

    logger.info("OOV rate: {:6.3f}% ({:6d}/{:6d})".format(oov_count / N * 100, oov_count, N))


if __name__ == '__main__':
    main()
