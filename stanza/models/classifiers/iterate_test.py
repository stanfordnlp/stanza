import argparse
import glob
import logging

import stanza.models.classifier as classifier
import stanza.models.classifiers.cnn_classifier as cnn_classifier
import stanza.models.classifiers.classifier_args as classifier_args
from stanza.models.common import utils
from stanza.models.common.pretrain import Pretrain

"""
A script for running the same test file on several different classifiers.

For each one, it will output the accuracy and, if possible, the confusion matrix.

Includes the arguments for pretrain, which allows for passing in a
different directory for the pretrain file.

Example command line:
  python3 -m stanza.models.classifiers.iterate_test  --test_file extern_data/sentiment/sst-processed/threeclass/test-threeclass-roots.txt --glob "saved_models/classifier/FC41_3class_en_ewt_FS*ACC66*"
"""

logger = logging.getLogger('stanza')

def parse_args():
    parser = argparse.ArgumentParser()

    classifier_args.add_common_args(parser)

    parser.add_argument('--test_file', type=str, default='extern_data/sentiment/sst-processed/binary/test-binary-roots.txt', help='Input file to use as the test set.')

    parser.add_argument('--glob', type=str, default='saved_models/classifier/*classifier*pt', help='Model file(s) to test.')

    args = parser.parse_args()
    return args

args = parse_args()
seed = utils.set_random_seed(args.seed, args.cuda)

model_files = []
for glob_piece in args.glob.split():
    model_files.extend(glob.glob(glob_piece))
model_files = sorted(set(model_files))

test_set = classifier.read_dataset(args.test_file, args.wordvec_type, min_len=None)
logger.info("Using test set: %s" % args.test_file)

pretrain = classifier.load_pretrain(args)

device = None
for load_name in model_files:
    logger.info("Testing %s" % load_name)
    model = cnn_classifier.load(load_name, pretrain)
    if args.cuda:
        model.cuda()
    if device is None:
        device = next(model.parameters()).device
        logger.info("Current device: %s" % device)

    labels = model.labels
    classifier.check_labels(labels, test_set)

    confusion = classifier.confusion_dataset(model, test_set, device=device)
    correct, total = classifier.confusion_to_accuracy(confusion)
    logger.info("  Results: %d correct of %d examples.  Accuracy: %f" %
                (correct, total, correct / total))
    logger.info("Confusion matrix:\n{}".format(classifier.format_confusion(confusion, model.labels)))
