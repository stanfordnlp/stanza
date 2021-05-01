"""
Entry point for training and evaluating a Bi-LSTM language identifier
"""

import argparse
import os

from datetime import datetime
from stanza.models.langid.data import DataLoader
from stanza.models.langid.trainer import Trainer
from tqdm import tqdm


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", help="batch size for training", type=int, default=64)
    parser.add_argument("--data-dir", help="directory with train/dev/test data", default=None)
    parser.add_argument("--mode", help="train or evaluate", default="train")
    parser.add_argument("--num-epochs", help="number of epochs for training", type=int, default=50)
    parser.add_argument("--randomize", help="take random substrings of samples", action="store_true")
    parser.add_argument("--save-name", help="where to save model", default=None)
    parser.add_argument("--use-gpu", help="whether to use gpu", type=bool, default=True)
    args = parser.parse_args(args=args)
    return args


def main(args=None):
    args = parse_args(args=args)
    if args.mode == "train":
        train(args)
    else:
        eval(args)


def build_indexes(args):
    tag_to_idx = {}
    char_to_idx = {}
    train_files = [f"{args.data_dir}/{x}" for x in os.listdir(args.data_dir) if "train" in x]
    for train_file in train_files(args):
        lines = open(f"{args.data_dir}/{train_file}").read().split("\n")
        examples = [json.loads(line) for line in lines if line.strip()]
        for example in examples:
            label = example["label"]
            if label not in tag_to_idx:
                tag_to_idx[label] = len(tag_to_idx)
            sequence = example["text"]
            for char in list(sequence):
                if char not in char_to_idx:
                    char_to_idx[char] = len(char_to_idx)
    char_to_idx["UNK"] = len(char_to_idx)
    char_to_idx["<PAD>"] = len(char_to_idx)

    return tag_to_idx, char_to_idx


def train(args):
    # set up indexes
    tag_to_idx, char_to_idx = build_indexes(args)
    # load training data
    train_data = DataLoader()
    train_files = [f"{args.data_dir}/{x}" for x in os.listdir(args.data_dir) if "train" in x]
    train_data.load_data(args.batch_size, train_files, char_to_idx, tag_to_idx, args.randomize)
    # load dev data
    dev_data = DataLoader()
    dev_files = [f"{args.data_dir}/{x}" for x in os.listdir(args.data_dir) if "dev" in x]
    dev_data.load_data(args.batch_size, dev_files, char_to_idx, tag_to_idx, randomize=False)
    # set up trainer
    trainer_config = {
        "model_path": args.save_name,
        "char_to_idx": char_to_idx,
        "tag_to_idx": tag_to_idx,
        "batch_size": args.batch_size,
        "lang_weights": train_data.lang_weights
    }
    trainer = Trainer(trainer_config, args.use_gpu)
    # run training
    best_accuracy = 0.0
    for epoch in range(1, args.num_epochs+1):
        print(f"{datetime.now()}\tEpoch {epoch}")
        print(f"{datetime.now()}\tNum training batches: {len(train_data.batches)}")
        for train_batch in tqdm(train_data.batches):
            inputs = (train_batch["sentences"], train_batch["targets"])
            trainer.update(inputs)
        print("Epoch complete. Evaluating on dev data.")
        total_correct = 0
        total_examples = 0
        for dev_batch in tqdm(dev_data.batches):
            inputs = (dev_batch["sentences"], dev_batch["targets"])
            predictions = trainer.predict(inputs)
            num_correct = torch.sum((predictions == dev_batch["targets"]).type(torch.long)).item()
            total_correct += num_correct
            total_examples += dev_batch["sentences"].size()[0]
        current_dev_accuracy = float(total_correct)/float(total_examples)
        print(f"Current dev accuracy: {current_accuracy}")
        if current_dev_accuracy > best_accuracy:
            trainer.save()


if __name__ == "__main__":
    main()








