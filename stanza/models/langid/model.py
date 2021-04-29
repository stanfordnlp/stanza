import torch
import torch.nn as nn


class LangIDBiLSTM(nn.Module):
    """
    Multi-layer BiLSTM model for language detecting. Based on \"A reproduction of Apple's bi-directional LSTM models
    for language identification in short strings.\" (Toftrup et al 2021)

    Arxiv: https://arxiv.org/abs/2102.06282
    GitHub: https://github.com/AU-DIS/LSTM_langid
    """

    def __init__(self, char_to_idx, tag_to_idx, num_layers, embedding_dim, hidden_dim, batch_size=64, weights=None):
        super(LangIDBiLSTM, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.char_to_idx = char_to_idx
        self.vocab_size = len(char_to_idx)
        self.tag_to_idx = tag_to_idx
        self.idx_to_tag = [i[1] for i in sorted([(v,k) for k,v in self.tag_to_idx.items()])]
        self.padding_idx = char_to_idx["<PAD>"]
        self.tagset_size = len(tag_to_idx)
        self.batch_size = batch_size
        self.loss_train = nn.CrossEntropyLoss(weight=weights)
        
        # embeddings for chars
        self.char_embeds = nn.Embedding(
                num_embeddings=self.vocab_size, 
                embedding_dim=self.embedding_dim,
                padding_idx=self.padding_idx
        )

        # the bidirectional LSTM
        self.lstm = nn.LSTM(
                self.embedding_dim, 
                self.hidden_dim,
                num_layers=self.num_layers,
                bidirectional=True,
                batch_first=True
        )

        # convert output to tag space
        self.hidden_to_tag = nn.Linear(
                self.hidden_dim * 2, 
                self.tagset_size
        )


    def loss(self, Y_hat, Y):
        return self.loss_train(Y_hat, Y)

    def forward(self, X):
        # embed input
        X = self.char_embeds(X)
        
        # run through LSTM
        X, _ = self.lstm(X)
        
        # run through linear layer
        X = self.hidden_to_tag(X)
        
        # sum character outputs for each sequence
        X = torch.sum(X, dim=1)

        return X

    def predict(self, X):
        label_idx = torch.argmax(self(X), dim=1).item()
        return self.idx_to_tag[label_idx]

    def save(self, path):
        """ Save a model at path """
        checkpoint = {
            "char_to_idx": self.char_to_idx,
            "tag_to_idx": self.tag_to_idx,
            "num_layers": self.num_layers,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "model_state_dict": self.state_dict()
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path, use_cuda=False, batch_size=64):
        """ Load a serialized model located at path """
        checkpoint = torch.load(path)
        weights = checkpoint["model_state_dict"]["loss_train.weight"]
        model = cls(checkpoint["char_to_idx"], checkpoint["tag_to_idx"], checkpoint["num_layers"],
                    checkpoint["embedding_dim"], checkpoint["hidden_dim"], batch_size=batch_size, weights=weights)
        model.load_state_dict(checkpoint["model_state_dict"])
        if use_cuda:
            model.to(torch.device("cuda"))
        return model

