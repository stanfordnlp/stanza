import argparse
import glob

import classifier
import classifier_args
from stanfordnlp.models.common import utils
from stanfordnlp.models.common.pretrain import Pretrain

def parse_args():
    parser = argparse.ArgumentParser()

    classifier_args.add_pretrain_args(parser)
    classifier_args.add_device_args(parser)

    parser.add_argument('--test_file', type=str, default='extern_data/sentiment/sst-processed/binary/test-binary-roots.txt', help='Input file to use as the test set.')

    parser.add_argument('--glob', type=str, default='saved_models/classifier/*classifier*pt', help='Model file(s) to test.')

    args = parser.parse_args()
    return args

args = parse_args()


model_files = []
for glob_piece in args.glob.split():
    model_files.extend(glob.glob(glob_piece))
model_files = sorted(set(model_files))

test_set = classifier.read_dataset(args.test_file, args.wordvec_type)
print("Using test set: %s" % args.test_file)

vec_file = utils.get_wordvec_file(args.wordvec_dir, args.shorthand)
pretrain_file = '{}/{}.pretrain.pt'.format(args.save_dir, args.shorthand)
pretrain = Pretrain(pretrain_file, vec_file, args.pretrain_max_vocab)
print("Embedding shape: %s" % str(pretrain.emb.shape))

device = None
for load_name in model_files:
    print("Testing %s" % load_name)
    model = classifier.load(load_name, pretrain)
    if args.cuda:
        model.cuda()
    if device is None:
        device = next(model.parameters()).device
        print("Current device: %s" % device)

    labels = model.labels
    classifier.check_labels(labels, test_set)

    correct = classifier.score_dataset(model, test_set, device=device)
    print("  Results: %d correct of %d examples.  Accuracy: %f" % 
          (correct, len(test_set), correct / len(test_set)))
