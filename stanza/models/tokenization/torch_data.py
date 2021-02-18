from torch.utils.data import Dataset
from stanza.models.tokenization.data import DataLoader
import torch


class DocsDataset(Dataset):
    def __init__(self, docs, transform):
        self.docs = docs
        self.transformtion = transform

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):

        text = self.docs[index].text
        units, features = self.transformtion(text)

        units = torch.LongTensor(units)
        features = torch.LongTensor(features)

        return units, features


class transform():
    """Implements function next from data.py (guess its preparing batches)"""
    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

    def __call__(self, text):
        d = DataLoader(self.config, input_text=text, vocab=self.vocab, evaluation=True)
        units, _, features, _ = d.next()
        units = units[0].numpy()
        features = features[0].numpy()

        return units, features