import json
import random
import torch


class DataLoader:
    """
    Class for loading language id data and providing batches

    Attempt to recreate data pre-processing from: https://github.com/AU-DIS/LSTM_langid

    Uses methods from: https://github.com/AU-DIS/LSTM_langid/blob/main/src/language_datasets.py

    Data format is same as LSTM_langid
    """

    def __init__(self, use_gpu=None):
        self.batches = None
        self.batches_iter = None
        self.tag_to_idx = None
        self.idx_to_tag = None
        self.lang_weights = None
        # set self.use_gpu and self.device
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = None

    def load_data(self, batch_size, data_files, char_index, tag_index, randomize=False, randomize_range=(5,20),
                  max_length=None):
        """
        Load sequence data and labels, calculate weights for weighted cross entropy loss.
        Data is stored in a file, 1 example per line
        Example: {"text": "Hello world.", "label": "en"}
        """

        # set up examples from data files
        examples = []
        for data_file in data_files:
            examples += [x for x in open(data_file).read().split("\n") if x.strip()]
        random.shuffle(examples)
        examples = [json.loads(x) for x in examples]

        # add additional labels in this data set to tag index
        tag_index = dict(tag_index)
        new_labels = set([x["label"] for x in examples]) - set(tag_index.keys())
        for new_label in new_labels:
            tag_index[new_label] = len(tag_index)
        self.tag_to_idx = tag_index
        self.idx_to_tag = [i[1] for i in sorted([(v,k) for k,v in self.tag_to_idx.items()])]
        
        # set up lang counts used for weights for cross entropy loss
        lang_counts = [0 for _ in tag_index]

        # optionally limit text to max length
        if max_length is not None:
            examples = [{"text": x["text"][:max_length], "label": x["label"]} for x in examples]

        # randomize data
        if randomize:
            split_examples = []
            for example in examples:
                sequence = example["text"]
                label = example["label"]
                sequences = DataLoader.randomize_data([sequence], upper_lim=randomize_range[1], 
                                                      lower_lim=randomize_range[0])
                split_examples += [{"text": seq, "label": label} for seq in sequences]
            examples = split_examples
            random.shuffle(examples)

        # break into equal length batches
        batch_lengths = {}
        for example in examples:
            sequence = example["text"]
            label = example["label"]
            if len(sequence) not in batch_lengths:
                batch_lengths[len(sequence)] = []
            sequence_as_list = [char_index.get(c, char_index["UNK"]) for c in list(sequence)]
            batch_lengths[len(sequence)].append((sequence_as_list, tag_index[label]))
            lang_counts[tag_index[label]] += 1
        for length in batch_lengths:
            random.shuffle(batch_lengths[length])

        # create final set of batches
        batches = []
        for length in batch_lengths:
            for sublist in [batch_lengths[length][i:i + batch_size] for i in
                            range(0, len(batch_lengths[length]), batch_size)]:
                batches.append(sublist)

        self.batches = [self.build_batch_tensors(batch) for batch in batches]

        # set up lang weights
        most_frequent = max(lang_counts)
        # set to 0.0 if lang_count is 0 or most_frequent/lang_count otherwise
        lang_counts = [(most_frequent * x)/(max(1, x) ** 2) for x in lang_counts]
        self.lang_weights = torch.tensor(lang_counts, device=self.device, dtype=torch.float)

        # shuffle batches to mix up lengths
        random.shuffle(self.batches)
        self.batches_iter = iter(self.batches)

    @staticmethod
    def randomize_data(sentences, upper_lim=20, lower_lim=5):
        """
        Takes the original data and creates random length examples with length between upper limit and lower limit
        From LSTM_langid: https://github.com/AU-DIS/LSTM_langid/blob/main/src/language_datasets.py
        """

        new_data = []
        for sentence in sentences:
            remaining = sentence
            while lower_lim < len(remaining):
                lim = random.randint(lower_lim, upper_lim)
                m = min(len(remaining), lim)
                new_sentence = remaining[:m]
                new_data.append(new_sentence)
                split = remaining[m:].split(" ", 1)
                if len(split) <= 1:
                    break
                remaining = split[1]
        random.shuffle(new_data)
        return new_data

    def build_batch_tensors(self, batch):
        """
        Helper to turn batches into tensors
        """

        batch_tensors = dict()
        batch_tensors["sentences"] = torch.tensor([s[0] for s in batch], device=self.device, dtype=torch.long)
        batch_tensors["targets"] = torch.tensor([s[1] for s in batch], device=self.device, dtype=torch.long)

        return batch_tensors

    def next(self):
        return next(self.batches_iter)

