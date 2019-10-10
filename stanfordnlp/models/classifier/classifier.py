import argparse
import ast
import collections
import logging
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import classifier_args
from stanfordnlp.models.common import utils
from stanfordnlp.models.common.pretrain import Pretrain

import cnn_classifier

logger = logging.getLogger(__name__)

#DEFAULT_TRAIN='extern_data/sentiment/sst-processed/binary/train-binary-phrases.txt'
#DEFAULT_DEV='extern_data/sentiment/sst-processed/binary/dev-binary-roots.txt'
#DEFAULT_TEST='extern_data/sentiment/sst-processed/binary/test-binary-roots.txt'

#DEFAULT_TRAIN='extern_data/sentiment/sst-processed/threeclass/train-3class-phrases.txt'
#DEFAULT_DEV='extern_data/sentiment/sst-processed/threeclass/dev-3class-roots.txt'
#DEFAULT_TEST='extern_data/sentiment/sst-processed/threeclass/test-3class-roots.txt'

DEFAULT_TRAIN='extern_data/sentiment/sst-processed/fiveclass/train-phrases.txt'
DEFAULT_DEV='extern_data/sentiment/sst-processed/fiveclass/dev-roots.txt'
DEFAULT_TEST='extern_data/sentiment/sst-processed/fiveclass/test-roots.txt'

"""A script for training and testing classifier models, especially on the SST.

If you run the script with no arguments, it will start trying to train
a sentiment model.

python stanfordnlp/models/classifier/classifier.py

This requires the sentiment dataset to be in an `extern_data`
directory, such as by symlinking it from somewhere else.

The default model is a CNN where the word vectors are first mapped to
channels with filters of a few different widths, those channels are
maxpooled over the entire sentence, and then the resulting pools have
fully connected layers until they reach the number of classes in the
training data.  You can see the defaults in the options below.

https://arxiv.org/abs/1408.5882

(Currently the CNN is the only sentence classifier implemented.)

You can train models with word vectors other than the default word2vec.  For example:

 nohup python -u stanfordnlp/models/classifier/classifier.py  --wordvec_type google --wordvec_dir extern_data/google --max_epochs 200 --filter_channels 1000 --fc_shapes 200,100 --base_name FC21_google > FC21_google.out 2>&1 &

A model trained on the 5 class dataset can be tested on the 2 class dataset with a command line like this:

python -u stanfordnlp/models/classifier/classifier.py  --wordvec_type google --wordvec_dir extern_data/google --no_train --load_name saved_models/classifier/FC21_google_en_ewt_FS_3_4_5_C_1000_FC_200_100_classifier.E0189-ACC45.87.pt --test_file extern_data/sentiment/sst-processed/binary/test-binary-roots.txt --test_remap_labels "{0:0, 1:0, 3:1, 4:1}"

A model trained on the 3 class dataset can be tested on the 2 class dataset with a command line like this:

python -u stanfordnlp/models/classifier/classifier.py  --wordvec_type google --wordvec_dir extern_data/google --no_train --load_name saved_models/classifier/FC21_3C_google_en_ewt_FS_3_4_5_C_1000_FC_200_100_classifier.E0101-ACC68.94.pt --test_file extern_data/sentiment/sst-processed/binary/test-binary-roots.txt --test_remap_labels "{0:0, 2:1}"

"""

def convert_fc_shapes(arg):
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
    parser = argparse.ArgumentParser()

    classifier_args.add_pretrain_args(parser)
    classifier_args.add_device_args(parser)

    parser.add_argument('--train', dest='train', default=True, action='store_true', help='Train the model (default)')
    parser.add_argument('--no_train', dest='train', action='store_false', help="Don't train the model")

    parser.add_argument('--load_name', type=str, default=None, help='Name for loading an existing model')

    parser.add_argument('--save_name', type=str, default=None, help='Name for saving the model')
    parser.add_argument('--base_name', type=str, default='sst', help="Base name of the model to use when building a model name from args")


    parser.add_argument('--train_file', type=str, default=DEFAULT_TRAIN, help='Input file to train a model from.  Each line is an example.  Should go <label> <tokenized sentence>.')
    parser.add_argument('--dev_file', type=str, default=DEFAULT_DEV, help='Input file to use as the dev set.')
    parser.add_argument('--test_file', type=str, default=DEFAULT_TEST, help='Input file to use as the test set.')
    parser.add_argument('--max_epochs', type=int, default=100)

    parser.add_argument('--filter_sizes', default=(3,4,5), type=ast.literal_eval, help='Filter sizes for the layer after the word vectors')
    parser.add_argument('--filter_channels', default=100, type=int, help='Number of channels for layers after the word vectors')
    parser.add_argument('--fc_shapes', default="100", type=convert_fc_shapes, help='Extra fully connected layers to put after the initial filters.  If set to blank, will FC directly from the max pooling to the output layer.')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout value to use')

    parser.add_argument('--seed', default=None, type=int, help='Random seed for model')

    parser.add_argument('--batch_size', default=50, type=int, help='Batch size when training')

    parser.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay (eg, l2 reg) to use in the optimizer')

    parser.add_argument('--optim', default='Adadelta', help='Optimizer type: SGD or Adadelta')

    parser.add_argument('--test_remap_labels', default=None, type=ast.literal_eval,
                        help='Map of which label each classifier label should map to.  For example, "{0:0, 1:0, 3:1, 4:1}" to map a 5 class sentiment test to a 2 class.  Any labels not mapped will be considered wrong')

    args = parser.parse_args()
    return args


# TODO: all this code is basically the same as for POS and NER.  Should refactor
def save(filename, model, args, skip_modules=True):
    model_state = model.state_dict()
    # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
    if skip_modules:
        skipped = [k for k in model_state.keys() if k.split('.')[0] in model.unsaved_modules]
        for k in skipped:
            del model_state[k]
    params = {
        'model': model_state,
        'config': model.config,
        'labels': model.labels,
    }
    try:
        torch.save(params, filename)
        logger.info("Model saved to {}".format(filename))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        logger.warning("Saving failed to {}... continuing anyway.  Error: {}".format(filename, e))

def print_config(config_name, config):
    print("-- %s --" % config_name)
    for k in config.__dict__:
        print("  --{}: {}".format(k, config.__dict__[k]))

def print_labels(labels):
    print("-- MODEL LABELS --")
    print("  {}".format(" ".join(labels)))

def load(filename, pretrain):
    try:
        checkpoint = torch.load(filename, lambda storage, loc: storage)
    except BaseException:
        logger.exception("Cannot load model from {}".format(filename))
        raise
    print("Loaded model {}".format(filename))
    print_labels(checkpoint['labels'])
    print_config("SAVED CONFIG", checkpoint['config'])
    model = cnn_classifier.CNNClassifier(pretrain.emb, pretrain.vocab, 
                                         checkpoint['labels'],
                                         checkpoint['config'])
    model.load_state_dict(checkpoint['model'], strict=False)
    return model


def update_text(sentence, wordvec_type):
    # TODO: this should be included in the model for when we are in a pipeline
    if wordvec_type == classifier_args.WVType.WORD2VEC:
        return sentence
    elif wordvec_type == classifier_args.WVType.GOOGLE:
        new_sentence = []
        for word in sentence:
            if word != '0' and word != '1':
                word = re.sub('[0-9]', '#', word)
            new_sentence.append(word)
        return new_sentence

def read_dataset(dataset, wordvec_type):
    """
    returns a list where the values of the list are
      label, [token...]
    TODO: make dataset items a class?
    """
    lines = open(dataset).readlines()
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if x]
    # stanford sentiment dataset has a lot of random - and /
    lines = [x.replace("-", " ") for x in lines]
    lines = [x.replace("/", " ") for x in lines]
    lines = [x.split() for x in lines]
    lines = [(x[0], update_text(x[1:], wordvec_type)) for x in lines]

    return lines

def dataset_labels(dataset):
    """
    Returns a sorted list of label name

    TODO: if everything is numeric, sort numerically?
    """
    labels = sorted(list(set([x[0] for x in dataset])))
    return labels

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


def score_dataset(model, dataset, label_map=None, device=None,
                  remap_labels=None):
    model.eval()
    if label_map is None:
        label_map = {x: y for (y, x) in enumerate(model.labels)}
    if device is None:
        device = next(model.parameters()).device
    correct = 0
    dataset_lengths = sort_dataset_by_len(dataset)
    
    for length in dataset_lengths.keys():
        batch = dataset_lengths[length]
        text = [x[1] for x in batch]
        expected_labels = [label_map[x[0]] for x in batch]

        output = model(text, device)

        # TODO: confusion matrix, etc
        for i in range(len(expected_labels)):
            predicted = torch.argmax(output[i])
            predicted_label = predicted.item()
            if remap_labels:
                if predicted_label in remap_labels:
                    predicted_label = remap_labels[predicted_label]
                else:
                    # if the label isn't something you're allow to predict, ocunt it wrong
                    continue
            if predicted_label == expected_labels[i]:
                correct = correct + 1
    return correct

def check_labels(labels, dataset):
    new_labels = dataset_labels(dataset)
    not_found = [i for i in new_labels if i not in labels]
    if not_found:
        raise RuntimeError('Found dev labels which do not exist in train:' + str(not_found))

def set_random_seed(seed, cuda):
    if seed is None:
        seed = random.randint(0, 1000000000)

    print("Using random seed: %d" % seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def checkpoint_name(filename, epoch, acc):
    """
    Build an informative checkpoint name from a base name, epoch #, and accuracy
    """
    root, ext = os.path.splitext(filename)
    return root + ".E{epoch:04d}-ACC{acc:05.2f}".format(**{"epoch": epoch, "acc": acc * 100}) + ext

def train_model(model, model_file, args, train_set, dev_set, labels):
    # TODO: separate this into a trainer like the other models.
    # TODO: possibly reuse the trainer code other models have
    # TODO: use a dataloader to possibly speed up the GPU usage
    # TODO different loss functions appropriate?
    loss_function = nn.CrossEntropyLoss()

    if args.cuda:
        loss_function.cuda()

    device = next(model.parameters()).device
    print("Current device: %s" % device)

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

    train_set_by_len = sort_dataset_by_len(train_set)

    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    batch_starts = list(range(0, len(train_set), args.batch_size))
    for epoch in range(args.max_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        shuffled = shuffle_dataset(train_set_by_len)
        model.train()
        random.shuffle(batch_starts)
        for batch_num, start_batch in enumerate(batch_starts):
            #print("Starting batch: %d" % start_batch)
            batch = shuffled[start_batch:start_batch+args.batch_size]
            text = [x[1] for x in batch]
            label = torch.stack([label_tensors[x[0]] for x in batch])

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(text, device)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if ((batch_num + 1) * args.batch_size) % 2000 < args.batch_size: # print every 2000 items
                print('[%d, %5d] average loss: %.3f' %
                      (epoch + 1, ((batch_num + 1) * args.batch_size), running_loss / 2000))
                epoch_loss += running_loss
                running_loss = 0.0

        correct = score_dataset(model, dev_set, label_map, device)
        print("Finished epoch %d.  Dev set: %d correct of %d examples.  Accuracy: %f  Total loss: %f" % 
              ((epoch + 1), correct, len(dev_set), correct / len(dev_set), epoch_loss))

        checkpoint_file = checkpoint_name(model_file, epoch + 1, correct / len(dev_set))
        save(checkpoint_file, model, args)
 
    save(model_file, model, args)

def load_pretrain(args):
    vec_file = utils.get_wordvec_file(args.wordvec_dir, args.shorthand)
    pretrain_file = '{}/{}.{}.pretrain.pt'.format(args.save_dir, args.shorthand, args.wordvec_type.name.lower())
    print("Loading pretrained embedding")
    pretrain = Pretrain(pretrain_file, vec_file, args.pretrain_max_vocab)
    print("Embedding shape: %s" % str(pretrain.emb.shape))
    return pretrain


def main():
    args = parse_args()
    set_random_seed(args.seed, args.cuda)

    utils.ensure_dir(args.save_dir)

    # TODO: maybe the dataset needs to be in a data loader in order to
    # make cuda operations faster
    train_set = read_dataset(args.train_file, args.wordvec_type)
    labels = dataset_labels(train_set)
    print("Using training set: %s" % args.train_file)
    print("Training set has %d labels" % len(labels))

    pretrain = load_pretrain(args)

    if args.load_name:
        model = load(args.load_name, pretrain)
    else:
        model = cnn_classifier.CNNClassifier(pretrain.emb, pretrain.vocab, labels, args)

    if args.cuda:
        model.cuda()

    print("Filter sizes: %s" % str(model.config.filter_sizes))
    print("Filter channels: %s" % str(model.config.filter_channels))
    print("Intermediate layers: %s" % str(model.config.fc_shapes))

    dev_set = read_dataset(args.dev_file, args.wordvec_type)
    print("Using dev set: %s" % args.dev_file)
    check_labels(model.labels, dev_set)

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
        train_model(model, model_file, args, train_set, dev_set, model.labels)

    test_set = read_dataset(args.test_file, args.wordvec_type)
    print("Using test set: %s" % args.test_file)
    check_labels(model.labels, test_set)

    correct = score_dataset(model, test_set, remap_labels=args.test_remap_labels)
    print("Test set: %d correct of %d examples.  Accuracy: %f" % 
          (correct, len(test_set), correct / len(test_set)))


if __name__ == '__main__':
    main()
