import argparse
import ast
import logging
import os
import random
import re
from enum import Enum

import torch
import torch.nn as nn

from stanza.models.common import loss
from stanza.models.common import utils
from stanza.models.pos.vocab import CharVocab

from stanza.models.classifiers.classifier_args import WVType, ExtraVectors
from stanza.models.classifiers.trainer import Trainer
import stanza.models.classifiers.data as data

from stanza.utils.confusion import format_confusion, confusion_to_accuracy, confusion_to_macro_f1


class Loss(Enum):
    CROSS = 1
    WEIGHTED_CROSS = 2
    LOG_CROSS = 3

class DevScoring(Enum):
    ACCURACY = 'ACC'
    WEIGHTED_F1 = 'WF'

logger = logging.getLogger('stanza')
tlogger = logging.getLogger('stanza.classifiers.trainer')

logging.getLogger('elmoformanylangs').setLevel(logging.WARNING)

DEFAULT_TRAIN='data/sentiment/en_sstplus.train.txt'
DEFAULT_DEV='data/sentiment/en_sst3roots.dev.txt'
DEFAULT_TEST='data/sentiment/en_sst3roots.test.txt'

"""A script for training and testing classifier models, especially on the SST.

If you run the script with no arguments, it will start trying to train
a sentiment model.

python3 -m stanza.models.classifier

This requires the sentiment dataset to be in an `extern_data`
directory, such as by symlinking it from somewhere else.

The default model is a CNN where the word vectors are first mapped to
channels with filters of a few different widths, those channels are
maxpooled over the entire sentence, and then the resulting pools have
fully connected layers until they reach the number of classes in the
training data.  You can see the defaults in the options below.

https://arxiv.org/abs/1408.5882

(Currently the CNN is the only sentence classifier implemented.)

To train with a more complicated CNN arch:

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 > FC41.out 2>&1 &

You can train models with word vectors other than the default word2vec.  For example:

 nohup python3 -u -m stanza.models.classifier  --wordvec_type google --wordvec_dir extern_data/google --max_epochs 200 --filter_channels 1000 --fc_shapes 200,100 --base_name FC21_google > FC21_google.out 2>&1 &

A model trained on the 5 class dataset can be tested on the 2 class dataset with a command line like this:

python3 -u -m stanza.models.classifier  --no_train --load_name saved_models/classifier/sst_en_ewt_FS_3_4_5_C_1000_FC_400_100_classifier.E0165-ACC41.87.pt --test_file data/sentiment/en_sst2roots.test.txt --test_remap_labels "{0:0, 1:0, 3:1, 4:1}"

python3 -u -m stanza.models.classifier  --wordvec_type google --wordvec_dir extern_data/google --no_train --load_name saved_models/classifier/FC21_google_en_ewt_FS_3_4_5_C_1000_FC_200_100_classifier.E0189-ACC45.87.pt --test_file data/sentiment/en_sst2roots.test.txt --test_remap_labels "{0:0, 1:0, 3:1, 4:1}"

A model trained on the 3 class dataset can be tested on the 2 class dataset with a command line like this:

python3 -u -m stanza.models.classifier  --wordvec_type google --wordvec_dir extern_data/google --no_train --load_name saved_models/classifier/FC21_3C_google_en_ewt_FS_3_4_5_C_1000_FC_200_100_classifier.E0101-ACC68.94.pt --test_file data/sentiment/en_sst2roots.test.txt --test_remap_labels "{0:0, 2:1}"

To train models on combined 3 class datasets:

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 --base_name FC41_3class  --extra_wordvec_method CONCAT --extra_wordvec_dim 200  --train_file data/sentiment/en_sstplus.train.txt --dev_file data/sentiment/en_sst3roots.dev.txt --test_file data/sentiment/en_sst3roots.test.txt > FC41_3class.out 2>&1 &

This tests that model:

python3 -u -m stanza.models.classifier --no_train --load_name en_sstplus.pt --test_file data/sentiment/en_sst3roots.test.txt

Here is an example for training a model in a different language:

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 --base_name FC41_german  --train_file data/sentiment/de_sb10k.train.txt --dev_file data/sentiment/de_sb10k.dev.txt --test_file data/sentiment/de_sb10k.test.txt --shorthand de_sb10k --min_train_len 3 --extra_wordvec_method CONCAT --extra_wordvec_dim 100 > de_sb10k.out 2>&1 &

This uses more data, although that wound up being worse for the German model:

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 --base_name FC41_german  --train_file data/sentiment/de_sb10k.train.txt,data/sentiment/de_scare.train.txt,data/sentiment/de_usage.train.txt --dev_file data/sentiment/de_sb10k.dev.txt --test_file data/sentiment/de_sb10k.test.txt --shorthand de_sb10k --min_train_len 3 --extra_wordvec_method CONCAT --extra_wordvec_dim 100 > de_sb10k.out 2>&1 &

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 --base_name FC41_chinese --train_file data/sentiment/zh_ren.train.txt --dev_file data/sentiment/zh_ren.dev.txt --test_file data/sentiment/zh_ren.test.txt --shorthand zh_ren --wordvec_type fasttext --extra_wordvec_method SUM --wordvec_pretrain_file ../stanza_resources/zh-hans/pretrain/gsdsimp.pt > zh_ren.out 2>&1 &

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 --save_name vi_vsfc.pt  --train_file data/sentiment/vi_vsfc.train.json --dev_file data/sentiment/vi_vsfc.dev.json --test_file data/sentiment/vi_vsfc.test.json --shorthand vi_vsfc --wordvec_pretrain_file ../stanza_resources/vi/pretrain/vtb.pt --wordvec_type word2vec --extra_wordvec_method SUM --dev_eval_scoring WEIGHTED_F1 > vi_vsfc.out 2>&1 &

python3 -u -m stanza.models.classifier --no_train --test_file extern_data/sentiment/vietnamese/_UIT-VSFC/test.txt --shorthand vi_vsfc --wordvec_pretrain_file ../stanza_resources/vi/pretrain/vtb.pt --wordvec_type word2vec --load_name vi_vsfc.pt
"""

def convert_fc_shapes(arg):
    """
    Returns a tuple of sizes to use in FC layers.

    For examples, converts "100" -> (100,)
    "100,200" -> (100,200)
    """
    arg = arg.strip()
    if not arg:
        return ()
    arg = ast.literal_eval(arg)
    if isinstance(arg, int):
        return (arg,)
    if isinstance(arg, tuple):
        return arg
    return tuple(arg)

# For the most part, these values are for the constituency parser.
# Only the WD for adadelta is originally for sentiment
# Also LR for adadelta and madgrad

# madgrad learning rate experiment on sstplus
# note that the hyperparameters are not cross-validated in tandem, so
# later changes may make some earlier experiments slightly out of date
# LR
#   0.01         failed to converge
#   0.004        failed to converge
#   0.003        0.5572
#   0.002        failed to converge
#   0.001        0.6857
#   0.0008       0.6799
#   0.0005       0.6849
#   0.00025      0.6749
#   0.0001       0.6746
#   0.00001      0.6536
#   0.000001     0.6267
# LR 0.001 produced the best model, but it does occasionally fail to
# converge to a working model, so we set the default to 0.0005 instead
DEFAULT_LEARNING_RATES = { "adamw": 0.0002, "adadelta": 1.0, "sgd": 0.001, "adabelief": 0.00005, "madgrad": 0.0005, "sgd": 0.001 }
DEFAULT_LEARNING_EPS = { "adabelief": 1e-12, "adadelta": 1e-6, "adamw": 1e-8 }
DEFAULT_LEARNING_RHO = 0.9
DEFAULT_MOMENTUM = { "madgrad": 0.9, "sgd": 0.9 }
DEFAULT_WEIGHT_DECAY = { "adamw": 0.05, "adadelta": 0.0001, "sgd": 0.01, "adabelief": 1.2e-6, "madgrad": 2e-6 }

def build_parser():
    """
    Build the argparse for the classifier.

    Refactored so that other utility scripts can use the same parser if needed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', dest='train', default=True, action='store_true', help='Train the model (default)')
    parser.add_argument('--no_train', dest='train', action='store_false', help="Don't train the model")

    parser.add_argument('--shorthand', type=str, default='en_ewt', help="Treebank shorthand, eg 'en' for English")

    parser.add_argument('--load_name', type=str, default=None, help='Name for loading an existing model')
    parser.add_argument('--save_dir', type=str, default='saved_models/classifier', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help='Name for saving the model')
    parser.add_argument('--base_name', type=str, default='sst', help="Base name of the model to use when building a model name from args")

    parser.add_argument('--checkpoint_save_name', type=str, default=None, help="File name to save the most recent checkpoint")
    parser.add_argument('--no_checkpoint', dest='checkpoint', action='store_false', help="Don't save checkpoints")

    parser.add_argument('--save_intermediate_models', default=False, action='store_true',
                        help='Save all intermediate models - this can be a lot!')

    parser.add_argument('--train_file', type=str, default=DEFAULT_TRAIN, help='Input file(s) to train a model from.  Each line is an example.  Should go <label> <tokenized sentence>.  Comma separated list.')
    parser.add_argument('--dev_file', type=str, default=DEFAULT_DEV, help='Input file(s) to use as the dev set.')
    parser.add_argument('--test_file', type=str, default=DEFAULT_TEST, help='Input file(s) to use as the test set.')
    parser.add_argument('--max_epochs', type=int, default=100)

    parser.add_argument('--filter_sizes', default=(3,4,5), type=ast.literal_eval, help='Filter sizes for the layer after the word vectors')
    parser.add_argument('--filter_channels', default=1000, type=ast.literal_eval, help='Number of channels for layers after the word vectors.  Int for same number of channels (scaled by width) for each filter, or tuple/list for exact lengths for each filter')
    parser.add_argument('--fc_shapes', default="400,100", type=convert_fc_shapes, help='Extra fully connected layers to put after the initial filters.  If set to blank, will FC directly from the max pooling to the output layer.')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout value to use')

    parser.add_argument('--batch_size', default=50, type=int, help='Batch size when training')
    parser.add_argument('--dev_eval_steps', default=100000, type=int, help='Run the dev set after this many train steps.  Set to 0 to only do it once per epoch')
    parser.add_argument('--dev_eval_scoring', type=lambda x: DevScoring[x.upper()], default=DevScoring.WEIGHTED_F1,
                        help=('Scoring method to use for choosing the best model.  Options: %s' %
                              " ".join(x.name for x in DevScoring)))

    parser.add_argument('--weight_decay', default=None, type=float, help='Weight decay (eg, l2 reg) to use in the optimizer')
    parser.add_argument('--learning_rate', default=None, type=float, help='Learning rate to use in the optimizer')
    parser.add_argument('--momentum', default=None, type=float, help='Momentum to use in the optimizer')

    parser.add_argument('--optim', default='adadelta', choices=['adadelta', 'madgrad', 'sgd'], help='Optimizer type: SGD, Adadelta, or madgrad.  Highly recommend to install madgrad and use that')

    parser.add_argument('--test_remap_labels', default=None, type=ast.literal_eval,
                        help='Map of which label each classifier label should map to.  For example, "{0:0, 1:0, 3:1, 4:1}" to map a 5 class sentiment test to a 2 class.  Any labels not mapped will be considered wrong')
    parser.add_argument('--forgive_unmapped_labels', dest='forgive_unmapped_labels', default=True, action='store_true',
                        help='When remapping labels, such as from 5 class to 2 class, pick a different label if the first guess is not remapped.')
    parser.add_argument('--no_forgive_unmapped_labels', dest='forgive_unmapped_labels', action='store_false',
                        help="When remapping labels, such as from 5 class to 2 class, DON'T pick a different label if the first guess is not remapped.")

    parser.add_argument('--loss', type=lambda x: Loss[x.upper()], default=Loss.CROSS,
                        help="Whether to use regular cross entropy or scale it by 1/log(quantity)")
    parser.add_argument('--min_train_len', type=int, default=0,
                        help="Filter sentences less than this length")

    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--wordvec_raw_file', type=str, default=None, help='Exact name of the raw wordvec file to read')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/wordvec', help='Directory of word vectors')
    parser.add_argument('--wordvec_type', type=lambda x: WVType[x.upper()], default='word2vec', help='Different vector types have different options, such as google 300d replacing numbers with #')
    parser.add_argument('--extra_wordvec_dim', type=int, default=0, help="Extra dim of word vectors - will be trained")
    parser.add_argument('--extra_wordvec_method', type=lambda x: ExtraVectors[x.upper()], default='sum', help='How to train extra dimensions of word vectors, if at all')
    parser.add_argument('--extra_wordvec_max_norm', type=float, default=None, help="Max norm for initializing the extra vectors")

    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")
    parser.add_argument('--charlm_projection', type=int, default=None, help="Project the charlm values to this dimension")
    parser.add_argument('--char_lowercase', dest='char_lowercase', action='store_true', help="Use lowercased characters in character model.")

    parser.add_argument('--elmo_model', default='extern_data/manyelmo/english', help='Directory with elmo model')
    parser.add_argument('--use_elmo', dest='use_elmo', default=False, action='store_true', help='Use an elmo model as a source of parameters')
    parser.add_argument('--elmo_projection', type=int, default=None, help='Project elmo to this many dimensions')

    parser.add_argument('--bert_model', type=str, default=None, help="Use an external bert model (requires the transformers package)")
    parser.add_argument('--no_bert_model', dest='bert_model', action="store_const", const=None, help="Don't use bert")

    parser.add_argument('--bilstm', dest='bilstm', action='store_true', default=True, help="Use a bilstm after the inputs, before the convs.  Using bilstm is about as accurate and significantly faster (because of dim reduction) than going straight to the filters")
    parser.add_argument('--no_bilstm', dest='bilstm', action='store_false', help="Don't use a bilstm after the inputs, before the convs.")
    # somewhere between 200-300 seems to be the sweet spot for a couple datasets:
    # dev set macro f1 scores on 3 class problems
    # note that these were only run once each
    # more trials might narrow down which ones works best
    # es_tass2020:
    #   150        0.5580
    #   200        0.5629
    #   250        0.5586
    #   300        0.5642    <---
    #   400        0.5525
    #   500        0.5579
    #   750        0.5585
    # en_sstplus:
    #   150        0.6816
    #   200        0.6721
    #   250        0.6915    <---
    #   300        0.6824
    #   400        0.6757
    #   500        0.6770
    #   750        0.6781
    # de_sb10k
    #   150        0.6745
    #   200        0.6798    <---
    #   250        0.6459
    #   300        0.6665
    #   400        0.6521
    #   500        0.6584
    #   750        0.6447
    parser.add_argument('--bilstm_hidden_dim', type=int, default=300, help="Dimension of the bilstm to use")

    parser.add_argument('--maxpool_width', type=int, default=1, help="Width of the maxpool kernel to use")

    parser.add_argument('--wandb', action='store_true', help='Start a wandb session and write the results of training.  Only applies to training.  Use --wandb_name instead to specify a name')
    parser.add_argument('--wandb_name', default=None, help='Name of a wandb session to start when training.  Will default to the dataset short name')

    parser.add_argument('--seed', default=None, type=int, help='Random seed for model')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA for training/testing', default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_false', help='Ignore CUDA.', dest='cuda')

    return parser

def parse_args(args=None):
    """
    Add arguments for building the classifier.
    Parses command line args and returns the result.
    """
    parser = build_parser()
    args = parser.parse_args(args)

    if args.wandb_name:
        args.wandb = True

    args.optim = args.optim.lower()
    if args.weight_decay is None:
        args.weight_decay = DEFAULT_WEIGHT_DECAY.get(args.optim, None)
    if args.momentum is None:
        args.momentum = DEFAULT_MOMENTUM.get(args.optim, None)
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LEARNING_RATES.get(args.optim, None)

    return args


def confusion_dataset(model, dataset):
    """
    Returns a confusion matrix

    First key: gold
    Second key: predicted
    so: confusion_matrix[gold][predicted]
    """
    model.eval()
    index_label_map = {x: y for (x, y) in enumerate(model.labels)}

    dataset_lengths = data.sort_dataset_by_len(dataset)

    confusion_matrix = {}
    for label in model.labels:
        confusion_matrix[label] = {}

    for length in dataset_lengths.keys():
        batch = dataset_lengths[length]
        text = [x[1] for x in batch]
        expected_labels = [x[0] for x in batch]

        output = model(text)
        for i in range(len(expected_labels)):
            predicted = torch.argmax(output[i])
            predicted_label = index_label_map[predicted.item()]
            confusion_matrix[expected_labels[i]][predicted_label] = confusion_matrix[expected_labels[i]].get(predicted_label, 0) + 1

    return confusion_matrix


def score_dataset(model, dataset, label_map=None,
                  remap_labels=None, forgive_unmapped_labels=False):
    """
    remap_labels: a dict from old label to new label to use when
    testing a classifier on a dataset with a simpler label set.
    For example, a model trained on 5 class sentiment can be tested
    on a binary distribution with {"0": "0", "1": "0", "3": "1", "4": "1"}

    forgive_unmapped_labels says the following: in the case that the
    model predicts "2" in the above example for remap_labels, instead
    treat the model's prediction as whichever label it gave the
    highest score
    """
    model.eval()
    if label_map is None:
        label_map = {x: y for (y, x) in enumerate(model.labels)}
    correct = 0
    dataset_lengths = data.sort_dataset_by_len(dataset)

    for length in dataset_lengths.keys():
        # TODO: possibly break this up into smaller batches
        batch = dataset_lengths[length]
        text = [x[1] for x in batch]
        expected_labels = [label_map[x[0]] for x in batch]

        output = model(text)

        for i in range(len(expected_labels)):
            predicted = torch.argmax(output[i])
            predicted_label = predicted.item()
            if remap_labels:
                if predicted_label in remap_labels:
                    predicted_label = remap_labels[predicted_label]
                else:
                    found = False
                    if forgive_unmapped_labels:
                        items = []
                        for j in range(len(output[i])):
                            items.append((output[i][j].item(), j))
                        items.sort(key=lambda x: -x[0])
                        for _, item in items:
                            if item in remap_labels:
                                predicted_label = remap_labels[item]
                                found = True
                                break
                    # if slack guesses allowed, none of the existing
                    # labels matched, so we count it wrong.  if slack
                    # guesses not allowed, just count it wrong
                    if not found:
                        continue

            if predicted_label == expected_labels[i]:
                correct = correct + 1
    return correct

def score_dev_set(model, dev_set, dev_eval_scoring):
    confusion_matrix = confusion_dataset(model, dev_set)
    logger.info("Dev set confusion matrix:\n{}".format(format_confusion(confusion_matrix, model.labels)))
    correct, total = confusion_to_accuracy(confusion_matrix)
    macro_f1 = confusion_to_macro_f1(confusion_matrix)
    logger.info("Dev set: %d correct of %d examples.  Accuracy: %f" %
                (correct, len(dev_set), correct / len(dev_set)))
    logger.info("Macro f1: {}".format(macro_f1))

    accuracy = correct / total
    if dev_eval_scoring is DevScoring.ACCURACY:
        return accuracy, accuracy, macro_f1
    elif dev_eval_scoring is DevScoring.WEIGHTED_F1:
        return macro_f1, accuracy, macro_f1
    else:
        raise ValueError("Unknown scoring method {}".format(dev_eval_scoring))

def intermediate_name(filename, epoch, dev_scoring, score):
    """
    Build an informative intermediate checkpoint name from a base name, epoch #, and accuracy
    """
    root, ext = os.path.splitext(filename)
    return root + ".E{epoch:04d}-{score_type}{acc:05.2f}".format(**{"epoch": epoch, "score_type": dev_scoring.value, "acc": score * 100}) + ext

def log_param_sizes(model):
    logger.debug("--- Model parameter sizes ---")
    total_size = 0
    for name, param in model.named_parameters():
        param_size = param.element_size() * param.nelement()
        total_size += param_size
        logger.debug("  %s %d %d %d", name, param.element_size(), param.nelement(), param_size)
    logger.debug("  Total size: %d", total_size)

def train_model(trainer, model_file, checkpoint_file, args, train_set, dev_set, labels):
    tlogger.setLevel(logging.DEBUG)

    # TODO: use a (torch) dataloader to possibly speed up the GPU usage
    model = trainer.model
    optimizer = trainer.optimizer

    device = next(model.parameters()).device
    logger.info("Current device: %s" % device)

    label_map = {x: y for (y, x) in enumerate(labels)}
    label_tensors = {x: torch.tensor(y, requires_grad=False, device=device)
                     for (y, x) in enumerate(labels)}

    if args.loss == Loss.CROSS:
        loss_function = nn.CrossEntropyLoss()
    elif args.loss == Loss.WEIGHTED_CROSS:
        loss_function = loss.weighted_cross_entropy_loss([label_map[x[0]] for x in train_set], log_dampened=False)
    elif args.loss == Loss.LOG_CROSS:
        loss_function = loss.weighted_cross_entropy_loss([label_map[x[0]] for x in train_set], log_dampened=True)
    else:
        raise ValueError("Unknown loss function {}".format(args.loss))
    loss_function.to(device)

    train_set_by_len = data.sort_dataset_by_len(train_set)

    if trainer.global_step > 0:
        # We reloaded the model, so let's report its current dev set score
        _ = score_dev_set(model, dev_set, args.dev_eval_scoring)
        logger.info("Reloaded model for continued training.")
        if trainer.best_score is not None:
            logger.info("Previous best score: %.5f", trainer.best_score)

    log_param_sizes(model)

    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    batch_starts = list(range(0, len(train_set), args.batch_size))

    if args.wandb:
        import wandb
        wandb_name = args.wandb_name if args.wandb_name else "%s_classifier" % args.shorthand
        wandb.init(name=wandb_name, config=args)
        wandb.run.define_metric('accuracy', summary='max')
        wandb.run.define_metric('macro_f1', summary='max')
        wandb.run.define_metric('epoch_loss', summary='min')

    for trainer.epochs_trained in range(trainer.epochs_trained, args.max_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        shuffled = data.shuffle_dataset(train_set_by_len)
        model.train()
        random.shuffle(batch_starts)
        for batch_num, start_batch in enumerate(batch_starts):
            trainer.global_step += 1
            logger.debug("Starting batch: %d step %d", start_batch, trainer.global_step)

            batch = shuffled[start_batch:start_batch+args.batch_size]
            text = [x[1] for x in batch]
            label = torch.stack([label_tensors[x[0]] for x in batch])

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(text)
            batch_loss = loss_function(outputs, label)
            batch_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += batch_loss.item()
            if ((batch_num + 1) * args.batch_size) % 2000 < args.batch_size: # print every 2000 items
                train_loss = running_loss / 2000
                logger.info('[%d, %5d] Average loss: %.3f' %
                            (trainer.epochs_trained + 1, ((batch_num + 1) * args.batch_size), train_loss))
                if args.wandb:
                    wandb.log({'train_loss': train_loss}, step=trainer.global_step)
                if (args.dev_eval_steps > 0 and
                    ((batch_num + 1) * args.batch_size) % args.dev_eval_steps < args.batch_size):
                    logger.info('---- Interim analysis ----')
                    dev_score, accuracy, macro_f1 = score_dev_set(model, dev_set, args.dev_eval_scoring)
                    if args.wandb:
                        wandb.log({'accuracy': accuracy, 'macro_f1': macro_f1}, step=trainer.global_step)
                    if trainer.best_score is None or dev_score > trainer.best_score:
                        trainer.best_score = dev_score
                        trainer.save(model_file, save_optimizer=False)
                        logger.info("Saved new best score model!  Accuracy %.5f   Macro F1 %.5f   Epoch %5d   Batch %d" % (accuracy, macro_f1, trainer.epochs_trained+1, batch_num+1))
                    model.train()
                epoch_loss += running_loss
                running_loss = 0.0
        # Add any leftover loss to the epoch_loss
        epoch_loss += running_loss

        logger.info("Finished epoch %d  Total loss %.3f" % (trainer.epochs_trained + 1, epoch_loss))
        dev_score, accuracy, macro_f1 = score_dev_set(model, dev_set, args.dev_eval_scoring)
        if args.wandb:
            wandb.log({'accuracy': accuracy, 'macro_f1': macro_f1, 'epoch_loss': epoch_loss}, step=trainer.global_step)
        if checkpoint_file:
            trainer.save(checkpoint_file, epochs_trained = trainer.epochs_trained + 1)
        if args.save_intermediate_models:
            intermediate_file = intermediate_name(model_file, trainer.epochs_trained + 1, args.dev_eval_scoring, dev_score)
            trainer.save(intermediate_file, save_optimizer=False)
        if trainer.best_score is None or dev_score > trainer.best_score:
            trainer.best_score = dev_score
            trainer.save(model_file, save_optimizer=False)
            logger.info("Saved new best score model!  Accuracy %.5f   Macro F1 %.5f   Epoch %5d" % (accuracy, macro_f1, trainer.epochs_trained+1))

    if args.wandb:
        wandb.finish()

def print_args(args):
    """
    For record keeping purposes, print out the arguments when training
    """
    args = vars(args)
    keys = sorted(args.keys())
    log_lines = ['%s: %s' % (k, args[k]) for k in keys]
    logger.info('ARGS USED AT TRAINING TIME:\n%s\n' % '\n'.join(log_lines))

def main(args=None):
    args = parse_args(args)
    seed = utils.set_random_seed(args.seed, args.cuda)
    logger.info("Using random seed: %d" % seed)

    utils.ensure_dir(args.save_dir)

    save_name = args.save_name
    if not(save_name):
        save_name = args.base_name + "_" + args.shorthand + "_"
        save_name = save_name + "FS_%s_" % "_".join([str(x) for x in args.filter_sizes])
        save_name = save_name + "C_%d_" % args.filter_channels
        if model.config.fc_shapes:
            save_name = save_name + "FC_%s_" % "_".join([str(x) for x in model.config.fc_shapes])
        save_name = save_name + "classifier.pt"

    # TODO: maybe the dataset needs to be in a torch data loader in order to
    # make cuda operations faster
    checkpoint_file = None
    if args.train:
        train_set = data.read_dataset(args.train_file, args.wordvec_type, args.min_train_len)
        logger.info("Using training set: %s" % args.train_file)
        logger.info("Training set has %d labels" % len(data.dataset_labels(train_set)))
        tlogger.setLevel(logging.DEBUG)

        if args.checkpoint:
            checkpoint_file = utils.checkpoint_name(args.save_dir, save_name, args.checkpoint_save_name)
    elif not args.load_name:
        if args.save_name:
            args.load_name = args.save_name
        else:
            raise ValueError("No model provided and not asked to train a model.  This makes no sense")
    else:
        train_set = None

    if args.train and checkpoint_file is not None and os.path.exists(checkpoint_file):
        trainer = Trainer.load(checkpoint_file, args, load_optimizer=args.train)
    elif args.load_name:
        trainer = Trainer.load(args.load_name, args, load_optimizer=args.train)
    else:
        trainer = Trainer.build_new_model(args, train_set)

    logger.info("Filter sizes: %s" % str(trainer.model.config.filter_sizes))
    logger.info("Filter channels: %s" % str(trainer.model.config.filter_channels))
    logger.info("Intermediate layers: %s" % str(trainer.model.config.fc_shapes))

    model_file = os.path.join(args.save_dir, save_name)

    if args.train:
        print_args(args)

        dev_set = data.read_dataset(args.dev_file, args.wordvec_type, min_len=None)
        logger.info("Using dev set: %s", args.dev_file)
        logger.info("Training set has %d items", len(train_set))
        logger.info("Dev set has %d items", len(dev_set))
        data.check_labels(trainer.model.labels, dev_set)

        train_model(trainer, model_file, checkpoint_file, args, train_set, dev_set, trainer.model.labels)

    test_set = data.read_dataset(args.test_file, args.wordvec_type, min_len=None)
    logger.info("Using test set: %s" % args.test_file)
    data.check_labels(trainer.model.labels, test_set)

    if args.test_remap_labels is None:
        confusion_matrix = confusion_dataset(trainer.model, test_set)
        logger.info("Confusion matrix:\n{}".format(format_confusion(confusion_matrix, trainer.model.labels)))
        correct, total = confusion_to_accuracy(confusion_matrix)
        logger.info("Macro f1: {}".format(confusion_to_macro_f1(confusion_matrix)))
    else:
        correct = score_dataset(trainer.model, test_set,
                                remap_labels=args.test_remap_labels,
                                forgive_unmapped_labels=args.forgive_unmapped_labels)
        total = len(test_set)
    logger.info("Test set: %d correct of %d examples.  Accuracy: %f" %
                (correct, total, correct / total))

if __name__ == '__main__':
    main()
