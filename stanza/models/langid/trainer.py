import torch
import torch.optim as optim

from stanza.models.langid.model import LangIDBiLSTM


class Trainer:

    DEFAULT_BATCH_SIZE = 64
    DEFAULT_LAYERS = 2
    DEFAULT_EMBEDDING_DIM = 150
    DEFAULT_HIDDEN_DIM = 150

    def __init__(self, config, load_model=False, use_gpu=None):
        self.model_path = config["model_path"]
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self.device = torch.device("cuda") if self.use_gpu else None
        self.batch_size = config.get("batch_size", Trainer.DEFAULT_BATCH_SIZE)
        if load_model:
            self.load(config["load_model"])
        else:
            self.model = LangIDBiLSTM(config["char_to_idx"], config["tag_to_idx"], Trainer.DEFAULT_LAYERS, 
                                      Trainer.DEFAULT_EMBEDDING_DIM,
                                      Trainer.DEFAULT_HIDDEN_DIM,
                                      batch_size=self.batch_size,
                                      weights=config["lang_weights"]).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters())

    def update(self, inputs):
        self.model.train()
        sentences, targets = inputs
        self.optimizer.zero_grad()
        y_hat = self.model.forward(sentences)
        loss = self.model.loss(y_hat, targets)
        loss.backward()
        self.optimizer.step()

    def predict(self, inputs):
        self.model.eval()
        sentences, targets = inputs
        return torch.argmax(self.model(sentences), dim=1)

    def save(self, label=None):
        # save a copy of model with label
        if label:
            self.model.save(f"{self.model_path[:-3]}-{label}.pt")
        self.model.save(self.model_path)

    def load(self, model_path=None):
        if not model_path:
            model_path = self.model_path
        self.model = LangIDBiLSTM.load(model_path, self.use_gpu, self.batch_size)

