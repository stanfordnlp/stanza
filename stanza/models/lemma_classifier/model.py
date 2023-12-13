import torch
import torch.nn as nn
import torch.optim as optim
import utils
import os
from constants import *
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer



# Define a custom model for your binary classifier
class LemmaClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embeddings, padding_idx = 0, **kwargs):
        super(LemmaClassifier, self).__init__()

        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        self.embedding_dim = embedding_dim

        # Embedding layer with GloVe embeddings
        self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)

        # Optionally, include charlm embeddings  
        self.use_charlm = kwargs.get("charlm")

        if self.use_charlm:
            if kwargs.get("charlm_forward_file") is None or not os.path.exists(kwargs.get("charlm_forward_file")):
                raise FileNotFoundError(f'Could not find forward character model: {kwargs.get("charlm_forward_file", "FILE_NOT_PROVIDED")}')
            if kwargs.get("charlm_backward_file") is None or not os.path.exists(kwargs.get("charlm_backward_file")):
                raise FileNotFoundError(f'Could not find backward character model: {kwargs.get("charlm_backward_file", "FILE_NOT_PROVIDED")}')
            add_unsaved_module('charmodel_forward', CharacterLanguageModel.load(kwargs.get("charlm_forward_file"), finetune=False))
            add_unsaved_module('charmodel_backward', CharacterLanguageModel.load(kwargs.get("charlm_backward_file"), finetune=False))
            self.embedding_dim += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
        
        # LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

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
        

        # # Charlm   TODO: How to get chars, charoffsets, charlens, and char_orig_idx. Also, do we have to pack? Also, can the append be the same as it is now?

        # TODO: fix this!!
        if self.use_charlm:
            char_reps_forward = self.charmodel_forward.build_char_representation(# sentence)
            char_reps_backward = self.charmodel_backward.build_char_representation(# sentence)
        
        embeddings = torch.cat((embedded, char_reps_forward, char_reps_backward), 1)
        print(embeddings, embeddings.shape)
        lstm_out, (hidden, _) = self.lstm(embeddings)

        # Extract the hidden state at the index of the token
        lstm_out = lstm_out[pos_index]

        # MLP forward pass
        output = self.mlp(lstm_out)
        return output
