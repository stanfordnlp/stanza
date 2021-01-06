import argparse
import ast
import collections
import logging
import os
import random
import re
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim

from stanza.models.common import loss
from stanza.models.common import utils
from stanza.models.common.char_model import CharacterLanguageModel
from stanza.models.common.vocab import PAD, PAD_ID, UNK, UNK_ID
from stanza.models.common.pretrain import Pretrain
from stanza.models.pos.vocab import CharVocab

import stanza.models.classifiers.classifier_args as classifier_args
import stanza.models.classifiers.cnn_classifier as cnn_classifier
import stanza.models.classifiers.data as data


class Loss(Enum):
    CROSS = 1
    WEIGHTED_CROSS = 2
    LOG_CROSS = 3

class DevScoring(Enum):
    ACCURACY = 'ACC'
    WEIGHTED_F1 = 'WF'

logger = logging.getLogger('stanza')

DEFAULT_TRAIN='extern_data/sentiment/sst-processed/fiveclass/train-phrases.txt'
DEFAULT_DEV='extern_data/sentiment/sst-processed/fiveclass/dev-roots.txt'
DEFAULT_TEST='extern_data/sentiment/sst-processed/fiveclass/test-roots.txt'

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

python3 -u -m stanza.models.classifier  --no_train --load_name saved_models/classifier/sst_en_ewt_FS_3_4_5_C_1000_FC_400_100_classifier.E0165-ACC41.87.pt --test_file extern_data/sentiment/sst-processed/binary/test-binary-roots.txt --test_remap_labels "{0:0, 1:0, 3:1, 4:1}"

python3 -u -m stanza.models.classifier  --wordvec_type google --wordvec_dir extern_data/google --no_train --load_name saved_models/classifier/FC21_google_en_ewt_FS_3_4_5_C_1000_FC_200_100_classifier.E0189-ACC45.87.pt --test_file extern_data/sentiment/sst-processed/binary/test-binary-roots.txt --test_remap_labels "{0:0, 1:0, 3:1, 4:1}"

A model trained on the 3 class dataset can be tested on the 2 class dataset with a command line like this:

python3 -u -m stanza.models.classifier  --wordvec_type google --wordvec_dir extern_data/google --no_train --load_name saved_models/classifier/FC21_3C_google_en_ewt_FS_3_4_5_C_1000_FC_200_100_classifier.E0101-ACC68.94.pt --test_file extern_data/sentiment/sst-processed/binary/test-binary-roots.txt --test_remap_labels "{0:0, 2:1}"

To train models on combined 3 class datasets:

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 --base_name FC41_3class  --extra_wordvec_method CONCAT --extra_wordvec_dim 200  --train_file extern_data/sentiment/sst-processed/threeclass/train-threeclass-phrases.txt,extern_data/sentiment/MELD/train.txt,extern_data/sentiment/slsd/train.txt,extern_data/sentiment/arguana/train.txt,extern_data/sentiment/airline/train.txt,extern_data/sentiment/sst-processed/threeclass/extra-train-threeclass-phrases.txt,extern_data/sentiment/sst-processed/threeclass/checked-extra-threeclass-phrases.txt --dev_file extern_data/sentiment/sst-processed/threeclass/dev-threeclass-roots.txt --test_file extern_data/sentiment/sst-processed/threeclass/test-threeclass-roots.txt > FC41_3class.out 2>&1 &

This tests that model:

python3 -u -m stanza.models.classifier --no_train --load_name en_sstplus.pt --test_file extern_data/sentiment/sst-processed/threeclass/test-threeclass-roots.txt

Here is an example for training a model in a different language:

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 --base_name FC41_german  --train_file extern_data/sentiment/german/sb-10k/train.txt,extern_data/sentiment/german/scare/train.txt,extern_data/sentiment/USAGE/de-train.txt --dev_file extern_data/sentiment/german/sb-10k/dev.txt --test_file extern_data/sentiment/german/sb-10k/test.txt --shorthand de_sb10k --min_train_len 3 --extra_wordvec_method CONCAT --extra_wordvec_dim 100 > de_sb10k.out 2>&1 &

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 --base_name FC41_chinese  --train_file extern_data/sentiment/chinese/RenCECps/train.txt --dev_file extern_data/sentiment/chinese/RenCECps/dev.txt --test_file extern_data/sentiment/chinese/RenCECps/test.txt --shorthand zh_ren --wordvec_type fasttext --extra_wordvec_method SUM > zh_ren.out 2>&1 &

nohup python3 -u -m stanza.models.classifier --max_epochs 400 --filter_channels 1000 --fc_shapes 400,100 --save_name vi_vsfc.pt  --train_file extern_data/sentiment/vietnamese/_UIT-VSFC/train.txt --dev_file extern_data/sentiment/vietnamese/_UIT-VSFC/dev.txt --test_file extern_data/sentiment/vietnamese/_UIT-VSFC/test.txt --shorthand vi_vsfc --wordvec_pretrain_file ../stanza_resources/vi/pretrain/vtb.pt --wordvec_type word2vec --extra_wordvec_method SUM --dev_eval_scoring WEIGHTED_F1 > vi_vsfc.out 2>&1 &

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

def parse_args():
    """
    Add arguments for building the classifier.
    Parses command line args and returns the result.
    """
    parser = argparse.ArgumentParser()

    classifier_args.add_common_args(parser)

    parser.add_argument('--train', dest='train', default=True, action='store_true', help='Train the model (default)')
    parser.add_argument('--no_train', dest='train', action='store_false', help="Don't train the model")

    parser.add_argument('--load_name', type=str, default=None, help='Name for loading an existing model')

    parser.add_argument('--save_name', type=str, default=None, help='Name for saving the model')
    parser.add_argument('--base_name', type=str, default='sst', help="Base name of the model to use when building a model name from args")

    parser.add_argument('--save_intermediate_models', default=False, action='store_true',
                        help='Save all intermediate models - this can be a lot!')

    parser.add_argument('--train_file', type=str, default=DEFAULT_TRAIN, help='Input file(s) to train a model from.  Each line is an example.  Should go <label> <tokenized sentence>.  Comma separated list.')
    parser.add_argument('--dev_file', type=str, default=DEFAULT_DEV, help='Input file(s) to use as the dev set.')
    parser.add_argument('--test_file', type=str, default=DEFAULT_TEST, help='Input file(s) to use as the test set.')
    parser.add_argument('--max_epochs', type=int, default=100)

    parser.add_argument('--filter_sizes', default=(3,4,5), type=ast.literal_eval, help='Filter sizes for the layer after the word vectors')
    parser.add_argument('--filter_channels', default=100, type=int, help='Number of channels for layers after the word vectors')
    parser.add_argument('--fc_shapes', default="100", type=convert_fc_shapes, help='Extra fully connected layers to put after the initial filters.  If set to blank, will FC directly from the max pooling to the output layer.')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout value to use')

    parser.add_argument('--batch_size', default=50, type=int, help='Batch size when training')
    parser.add_argument('--dev_eval_steps', default=None, type=int, help='Run the dev set after this many train steps')
    parser.add_argument('--dev_eval_scoring', type=lambda x: DevScoring[x.upper()], default=DevScoring.ACCURACY,
                        help=('Scoring method to use for choosing the best model.  Options: %s' %
                              " ".join(x.name for x in DevScoring)))

    parser.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay (eg, l2 reg) to use in the optimizer')

    parser.add_argument('--optim', default='Adadelta', help='Optimizer type: SGD or Adadelta')

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

    parser.add_argument('--charlm', action='store_true', help="Turn on contextualized char embedding using pretrained character-level language model.")
    parser.add_argument('--charlm_save_dir', type=str, default='saved_models/charlm', help="Root dir for pretrained character-level language model.")
    parser.add_argument('--charlm_shorthand', type=str, default=None, help="Shorthand for character-level language model training corpus.")
    parser.add_argument('--charlm_projection', type=int, default=None, help="Project the charlm values to this dimension")
    parser.add_argument('--char_lowercase', dest='char_lowercase', action='store_true', help="Use lowercased characters in character model.")

    args = parser.parse_args()

    if args.charlm_shorthand is not None:
        args.charlm = True

    return args


def dataset_labels(dataset):
    """
    Returns a sorted list of label name
    """
    labels = set([x[0] for x in dataset])
    if all(re.match("^[0-9]+$", label) for label in labels):
        # if all of the labels are integers, sort numerically
        # maybe not super important, but it would be nicer than having
        # 10 before 2
        labels = [str(x) for x in sorted(map(int, list(labels)))]
    else:
        labels = sorted(list(labels))
    return labels

def dataset_vocab(dataset):
    vocab = set()
    for line in dataset:
        for word in line[1]:
            vocab.add(word)
    vocab = [PAD, UNK] + list(vocab)
    if vocab[PAD_ID] != PAD or vocab[UNK_ID] != UNK:
        raise ValueError("Unexpected values for PAD and UNK!")
    return vocab

def sort_dataset_by_len(dataset):
    """
    returns a dict mapping length -> list of items of that length
    an OrderedDict is used to that the mapping is sorted from smallest to largest
    """
    sorted_dataset = collections.OrderedDict()
    lengths = sorted(list(set(len(x[1]) for x in dataset)))
    for l in lengths:
        sorted_dataset[l] = []
    for item in dataset:
        sorted_dataset[len(item[1])].append(item)
    return sorted_dataset

def shuffle_dataset(sorted_dataset):
    """
    Given a dataset sorted by len, sorts within each length to make
    chunks of roughly the same size.  Returns all items as a single list.
    """
    dataset = []
    for l in sorted_dataset.keys():
        items = list(sorted_dataset[l])
        random.shuffle(items)
        dataset.extend(items)
    return dataset

def confusion_dataset(model, dataset, device=None):
    """
    Returns a confusion matrix

    First key: gold
    Second key: predicted
    so: confusion[gold][predicted]
    """
    model.eval()
    index_label_map = {x: y for (x, y) in enumerate(model.labels)}
    if device is None:
        device = next(model.parameters()).device

    dataset_lengths = sort_dataset_by_len(dataset)

    confusion = {}
    for label in model.labels:
        confusion[label] = {}

    for length in dataset_lengths.keys():
        batch = dataset_lengths[length]
        text = [x[1] for x in batch]
        expected_labels = [x[0] for x in batch]

        output = model(text, device)
        for i in range(len(expected_labels)):
            predicted = torch.argmax(output[i])
            predicted_label = index_label_map[predicted.item()]
            confusion[expected_labels[i]][predicted_label] = confusion[expected_labels[i]].get(predicted_label, 0) + 1

    return confusion


def confusion_to_accuracy(confusion):
    """
    Given a confusion dictionary returned by confusion_dataset, return correct, total
    """
    correct = 0
    total = 0
    for l1 in confusion.keys():
        for l2 in confusion[l1].keys():
            if l1 == l2:
                correct = correct + confusion[l1][l2]
            else:
                total = total + confusion[l1][l2]
    return correct, (correct + total)

def confusion_to_macro_f1(confusion):
    """
    Return the macro f1 for a confusion matrix.
    """
    keys = set()
    for k in confusion.keys():
        keys.add(k)
        for k2 in confusion.get(k).keys():
            keys.add(k2)

    sum_f1 = 0
    for k in keys:
        tp = 0
        fn = 0
        fp = 0
        for k2 in keys:
            if k == k2:
                tp = confusion.get(k, {}).get(k, 0)
            else:
                fn = fn + confusion.get(k, {}).get(k2, 0)
                fp = fp + confusion.get(k2, {}).get(k, 0)
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        sum_f1 = sum_f1 + f1

    return sum_f1 / len(keys)


def format_confusion(confusion, labels, hide_zeroes=False):
    """
    pretty print for confusion matrixes
    adapted from https://gist.github.com/zachguo/10296432
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    header = "    " + fst_empty_cell + " "

    for label in labels:
        header = header + "%{0}s ".format(columnwidth) % label
    text = [header]

    # Print rows
    for i, label1 in enumerate(labels):
        row = "    %{0}s ".format(columnwidth) % label1
        for j, label2 in enumerate(labels):
            confusion_cell = confusion.get(label1, {}).get(label2, 0)
            cell = "%{0}.1f".format(columnwidth) % confusion_cell
            if hide_zeroes:
                cell = cell if confusion_cell else empty_cell
            row = row + cell + " "
        text.append(row)
    return "\n".join(text)


def score_dataset(model, dataset, label_map=None, device=None,
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
    if device is None:
        device = next(model.parameters()).device
    correct = 0
    dataset_lengths = sort_dataset_by_len(dataset)

    for length in dataset_lengths.keys():
        # TODO: possibly break this up into smaller batches
        batch = dataset_lengths[length]
        text = [x[1] for x in batch]
        expected_labels = [label_map[x[0]] for x in batch]

        output = model(text, device)

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
    confusion = confusion_dataset(model, dev_set)
    logger.info("Dev set confusion matrix:\n{}".format(format_confusion(confusion, model.labels)))
    correct, total = confusion_to_accuracy(confusion)
    macro_f1 = confusion_to_macro_f1(confusion)
    logger.info("Dev set: %d correct of %d examples.  Accuracy: %f" %
                (correct, len(dev_set), correct / len(dev_set)))
    logger.info("Macro f1: {}".format(macro_f1))
    if dev_eval_scoring is DevScoring.ACCURACY:
        return correct / total
    elif dev_eval_scoring is DevScoring.WEIGHTED_F1:
        return macro_f1
    else:
        raise ValueError("Unknown scoring method {}".format(dev_eval_scoring))

def check_labels(labels, dataset):
    """
    Check that all of the labels in the dataset are in the known labels.
    Actually, unknown labels could be acceptable if we just treat the model as always wrong.
    However, this is a good sanity check to make sure the datasets match
    """
    new_labels = dataset_labels(dataset)
    not_found = [i for i in new_labels if i not in labels]
    if not_found:
        raise RuntimeError('Dataset contains labels which the model does not know about:' + str(not_found))

def checkpoint_name(filename, epoch, dev_scoring, score):
    """
    Build an informative checkpoint name from a base name, epoch #, and accuracy
    """
    root, ext = os.path.splitext(filename)
    return root + ".E{epoch:04d}-{score_type}{acc:05.2f}".format(**{"epoch": epoch, "score_type": dev_scoring.value, "acc": score * 100}) + ext

def train_model(model, model_file, args, train_set, dev_set, labels):
    # TODO: separate this into a trainer like the other models.
    # TODO: possibly reuse the trainer code other models have
    # TODO: use a (torch) dataloader to possibly speed up the GPU usage
    device = next(model.parameters()).device
    logger.info("Current device: %s" % device)

    # TODO: if reloading a model for continued training, the internal
    # parameters for the optimizer should be reloaded as well
    # Otherwise this ability is actually not very useful
    if args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                              weight_decay=args.weight_decay)
    elif args.optim.lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), weight_decay=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer: %s" % args.optim)

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
    if args.cuda:
        loss_function.cuda()

    train_set_by_len = sort_dataset_by_len(train_set)

    if args.load_name:
        # We reloaded the model, so let's report its current dev set score
        correct = score_dev_set(model, dev_set, args.dev_eval_scoring)
        logger.info("Reloaded model for continued training.")

    best_score = 0

    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    batch_starts = list(range(0, len(train_set), args.batch_size))
    for epoch in range(args.max_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        shuffled = shuffle_dataset(train_set_by_len)
        model.train()
        random.shuffle(batch_starts)
        for batch_num, start_batch in enumerate(batch_starts):
            logger.debug("Starting batch: %d" % start_batch)
            batch = shuffled[start_batch:start_batch+args.batch_size]
            text = [x[1] for x in batch]
            label = torch.stack([label_tensors[x[0]] for x in batch])

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(text, device)
            batch_loss = loss_function(outputs, label)
            batch_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += batch_loss.item()
            if ((batch_num + 1) * args.batch_size) % 2000 < args.batch_size: # print every 2000 items
                if (args.dev_eval_steps and
                    ((batch_num + 1) * args.batch_size) % args.dev_eval_steps < args.batch_size):
                    logger.info('[%d, %5d] Interim analysis' % (epoch + 1, ((batch_num + 1) * args.batch_size)))
                    dev_score = score_dev_set(model, dev_set, args.dev_eval_scoring)
                    if best_score is None or dev_score > best_score:
                        best_score = dev_score
                        cnn_classifier.save(model_file, model)
                        logger.info("Saved new best score model!")
                    model.train()
                else:
                    logger.info('[%d, %5d] Average loss: %.3f' %
                                (epoch + 1, ((batch_num + 1) * args.batch_size), running_loss / 2000))
                epoch_loss += running_loss
                running_loss = 0.0
        # Add any leftover loss to the epoch_loss
        epoch_loss += running_loss

        logger.info("Finished epoch %d" % (epoch + 1))
        dev_score = score_dev_set(model, dev_set, args.dev_eval_scoring)
        if args.save_intermediate_models:
            checkpoint_file = checkpoint_name(model_file, epoch + 1, args.dev_eval_scoring, dev_score)
            cnn_classifier.save(checkpoint_file, model)
        if best_score is None or dev_score > best_score:
            best_score = dev_score
            cnn_classifier.save(model_file, model)
            logger.info("Saved new best score model!")



def load_pretrain(args):
    if args.wordvec_pretrain_file:
        pretrain_file = args.wordvec_pretrain_file
    elif args.wordvec_type:
        pretrain_file = '{}/{}.{}.pretrain.pt'.format(args.save_dir, args.shorthand, args.wordvec_type.name.lower())
    else:
        raise Exception("TODO: need to get the wv type back from get_wordvec_file")

    logger.info("Looking for pretrained vectors in {}".format(pretrain_file))
    if os.path.exists(pretrain_file):
        vec_file = None
    elif args.wordvec_raw_file:
        vec_file = args.wordvec_raw_file
        logger.info("Pretrain not found.  Looking in {}".format(vec_file))
    else:
        vec_file = utils.get_wordvec_file(args.wordvec_dir, args.shorthand, args.wordvec_type.name.lower())
        logger.info("Pretrain not found.  Looking in {}".format(vec_file))
    pretrain = Pretrain(pretrain_file, vec_file, args.pretrain_max_vocab)
    logger.info("Embedding shape: %s" % str(pretrain.emb.shape))
    return pretrain


def print_args(args):
    """
    For record keeping purposes, print out the arguments when training
    """
    args = vars(args)
    keys = sorted(args.keys())
    log_lines = ['%s: %s' % (k, args[k]) for k in keys]
    logger.info('ARGS USED AT TRAINING TIME:\n%s\n' % '\n'.join(log_lines))

def main():
    args = parse_args()
    seed = utils.set_random_seed(args.seed, args.cuda)
    logger.info("Using random seed: %d" % seed)

    utils.ensure_dir(args.save_dir)

    # TODO: maybe the dataset needs to be in a torch data loader in order to
    # make cuda operations faster
    if args.train:
        train_set = data.read_dataset(args.train_file, args.wordvec_type, args.min_train_len)
        logger.info("Using training set: %s" % args.train_file)
        logger.info("Training set has %d labels" % len(dataset_labels(train_set)))
    elif not args.load_name:
        raise ValueError("No model provided and not asked to train a model.  This makes no sense")
    else:
        train_set = None

    pretrain = load_pretrain(args)

    if args.charlm:
        if args.charlm_shorthand is None:
            raise ValueError("CharLM Shorthand is required for loading pretrained CharLM model...")
        logger.info('Using pretrained contextualized char embedding')
        charlm_forward_file = '{}/{}_forward_charlm.pt'.format(args.charlm_save_dir, args.charlm_shorthand)
        charlm_backward_file = '{}/{}_backward_charlm.pt'.format(args.charlm_save_dir, args.charlm_shorthand)
        charmodel_forward = CharacterLanguageModel.load(charlm_forward_file, finetune=False)
        charmodel_backward = CharacterLanguageModel.load(charlm_backward_file, finetune=False)
    else:
        charmodel_forward = None
        charmodel_backward = None

    if args.load_name:
        model = cnn_classifier.load(args.load_name, pretrain,
                                    charmodel_forward, charmodel_backward)
    else:
        assert train_set is not None
        labels = dataset_labels(train_set)
        extra_vocab = dataset_vocab(train_set)
        model = cnn_classifier.CNNClassifier(pretrain=pretrain,
                                             extra_vocab=extra_vocab,
                                             labels=labels,
                                             charmodel_forward=charmodel_forward,
                                             charmodel_backward=charmodel_backward,
                                             args=args)

    if args.cuda:
        model.cuda()

    logger.info("Filter sizes: %s" % str(model.config.filter_sizes))
    logger.info("Filter channels: %s" % str(model.config.filter_channels))
    logger.info("Intermediate layers: %s" % str(model.config.fc_shapes))

    save_name = args.save_name
    if not(save_name):
        save_name = args.base_name + "_" + args.shorthand + "_"
        save_name = save_name + "FS_%s_" % "_".join([str(x) for x in model.config.filter_sizes])
        save_name = save_name + "C_%d_" % model.config.filter_channels
        if model.config.fc_shapes:
            save_name = save_name + "FC_%s_" % "_".join([str(x) for x in model.config.fc_shapes])
        save_name = save_name + "classifier.pt"
    model_file = os.path.join(args.save_dir, save_name)

    if args.train:
        print_args(args)

        dev_set = data.read_dataset(args.dev_file, args.wordvec_type, min_len=None)
        logger.info("Using dev set: %s" % args.dev_file)
        check_labels(model.labels, dev_set)

        train_model(model, model_file, args, train_set, dev_set, model.labels)

    test_set = data.read_dataset(args.test_file, args.wordvec_type, min_len=None)
    logger.info("Using test set: %s" % args.test_file)
    check_labels(model.labels, test_set)

    if args.test_remap_labels is None:
        confusion = confusion_dataset(model, test_set)
        logger.info("Confusion matrix:\n{}".format(format_confusion(confusion, model.labels)))
        correct, total = confusion_to_accuracy(confusion)
        logger.info("Macro f1: {}".format(confusion_to_macro_f1(confusion)))
    else:
        correct = score_dataset(model, test_set,
                                remap_labels=args.test_remap_labels,
                                forgive_unmapped_labels=args.forgive_unmapped_labels)
        total = len(test_set)
    logger.info("Test set: %d correct of %d examples.  Accuracy: %f" %
                (correct, total, correct / total))

if __name__ == '__main__':
    main()
