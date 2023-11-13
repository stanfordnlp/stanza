import torch
import torch.nn as nn
import torch.optim as optim
import utils
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer
from constants import *


# Define a custom model for your binary classifier
class LemmaClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embeddings, padding_idx = 0):
        super(LemmaClassifier, self).__init__()

        self.embedding_dim = embedding_dim

        # Embedding layer with GloVe embeddings
        self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, text: torch.tensor, pos_index: int):
        """
        Computes the forward pass of the neural net

        Args:
            text (torch.tensor): Tensor of the tokenized indices of the words in the input sentence, with unknown words having their index set to UNKNOWN_TOKEN_IDX
            pos_index (int): The position index of the target token for lemmatization classification in the sentence.

        Returns:
            torch.tensor: Output logits of the neural network
        """
        # Token embeddings
        glove = get_glove(self.embedding_dim)
        # UNKNOWN_TOKEN will be our <UNK> token
        # UNKNOWN_TOKEN_IDX will be the custom index for the <UNK> token
        unk_token_indices = utils.extract_unknown_token_indices(text, UNKNOWN_TOKEN_IDX)
        unknown_mask = (text == UNKNOWN_TOKEN_IDX)
        masked_indices = text.masked_fill(unknown_mask, 0)  # Replace UNKNOWN_TOKEN_IDX with 0 for embedding lookup

        # replace 0 token vectors with the true unknown 
        embedded = self.embedding(masked_indices)
        for unk_token_idx in unk_token_indices:
            embedded[unk_token_idx] = glove[UNKNOWN_TOKEN]

        lstm_out, (hidden, _) = self.lstm(embedded)

        # Extract the hidden state at the index of the token
        lstm_out = lstm_out[pos_index]

        # MLP forward pass
        output = self.mlp(lstm_out)
        return output
