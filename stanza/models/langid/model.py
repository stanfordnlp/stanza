import torch
import torch.nn as nn


class LangIDBiLSTM(nn.Module):
    """
    Multi-layer BiLSTM model for language detecting. A recreation of "A reproduction of Apple's bi-directional LSTM models
    for language identification in short strings." (Toftrup et al 2021)

    Arxiv: https://arxiv.org/abs/2102.06282
    GitHub: https://github.com/AU-DIS/LSTM_langid

    This class is similar to https://github.com/AU-DIS/LSTM_langid/blob/main/src/LSTMLID.py
    """

    def __init__(self, char_to_idx, tag_to_idx, num_layers, embedding_dim, hidden_dim, batch_size=64, weights=None, 
                 dropout=0.0, lang_subset=None):
        super(LangIDBiLSTM, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.char_to_idx = char_to_idx
        self.vocab_size = len(char_to_idx)
        self.tag_to_idx = tag_to_idx
        self.idx_to_tag = [i[1] for i in sorted([(v,k) for k,v in self.tag_to_idx.items()])]
        self.lang_subset = lang_subset
        self.padding_idx = char_to_idx["<PAD>"]
        self.tagset_size = len(tag_to_idx)
        self.batch_size = batch_size
        self.loss_train = nn.CrossEntropyLoss(weight=weights)
        self.dropout_prob = dropout
        
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

        # dropout layer
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def build_lang_mask(self, use_gpu=None):
        """
        Build language mask if a lang subset is specified (e.g. ["en", "fr"])

        The mask will be added to the results to set the prediction scores of illegal languages to -inf
        """
        device = torch.device("cuda") if use_gpu else None
        if self.lang_subset:
            lang_mask_list = [0.0 if lang in self.lang_subset else -float('inf') for lang in self.idx_to_tag]
            self.lang_mask = torch.tensor(lang_mask_list, device=device, dtype=torch.float)
        else:
            self.lang_mask = torch.zeros(len(self.idx_to_tag), device=device, dtype=torch.float)

    def loss(self, Y_hat, Y):
        return self.loss_train(Y_hat, Y)

    def forward(self, x):
        # embed input
        x = self.char_embeds(x)
        
        # run through LSTM
        x, _ = self.lstm(x)
        
        # run through linear layer
        x = self.hidden_to_tag(x)
        
        # sum character outputs for each sequence
        x = torch.sum(x, dim=1)

        return x

    def prediction_scores(self, x):
        prediction_probs = self(x)
        if self.lang_subset:
            prediction_batch_size = prediction_probs.size()[0]
            batch_mask = torch.stack([self.lang_mask for _ in range(prediction_batch_size)])
            prediction_probs = prediction_probs + batch_mask
        return torch.argmax(prediction_probs, dim=1)

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
    def load(cls, path, use_cuda=False, batch_size=64, lang_subset=None):
        """ Load a serialized model located at path """
        if use_cuda:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        weights = checkpoint["model_state_dict"]["loss_train.weight"]
        model = cls(checkpoint["char_to_idx"], checkpoint["tag_to_idx"], checkpoint["num_layers"],
                    checkpoint["embedding_dim"], checkpoint["hidden_dim"], batch_size=batch_size, weights=weights,
                    lang_subset=lang_subset)
        model.load_state_dict(checkpoint["model_state_dict"])
        if use_cuda:
            model.to(torch.device("cuda"))
        model.build_lang_mask(use_gpu=use_cuda)
        return model

