import torch
import torch.optim as optim

from stanza.models.langid.model import LangIDBiLSTM


class Trainer:

    DEFAULT_BATCH_SIZE = 64
    DEFAULT_LAYERS = 2
    DEFAULT_EMBEDDING_DIM = 150
    DEFAULT_HIDDEN_DIM = 150

    def __init__(self, config, use_gpu=None):
        self.model_path = config["model_path"]
        self.use_gpu = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.device = torch.device("cuda") if self.use_gpu else None
        self.batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
        self.model = LangIDBiLSTM(config["char_to_idx"], config["tag_to_idx"], DEFAULT_LAYERS, DEFAULT_EMBEDDING_DIM,
                                  DEFAULT_HIDDEN_DIM,
                                  batch_size=self.batch_size,
                                  weights=config["lang_weights"]).to(self.device)
        self.optimizer = optim.AdamW(model.parameters())

    def update(self, inputs):
        self.model.train()
        sentences, targets = inputs
        self.optimizer.zero_grad()
        y_hat = model.forward(sentences)
        loss = model.loss(y_hat, targets)
        loss.backward()
        self.optimizer.step()

    def predict(self, inputs):
        self.model.eval()
        sentences, targets = inputs
        return torch.argmax(self.model(sentences), dim=1)

    def save(self):
        self.model.save(self.model_path)

    def load(self):
        self.model = LangIDBiLSTM.load(self.model_path, self.use_gpu, self.batch_size)






