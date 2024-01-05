import torch
import torch.nn as nn
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from typing import List, Tuple

from stanza.models.common.vocab import UNK_ID
from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.constants import ModelType

class LemmaClassifierLSTM(LemmaClassifier):
    """
    Model architecture:
        Extracts word embeddings over the sentence, passes embeddings into a bi-LSTM to get a sentence encoding.
        From the LSTM output, we get the embedding fo the specific token that we classify on. That embedding 
        is fed into an MLP for classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab_map, pt_embedding, label_decoder, **kwargs):
        """
        Args:
            vocab_size (int): Size of the vocab being used (if custom vocab)
            embeddings (word vectors for embedding): What word embeddings to use (currently only supports GloVe) TODO add more!
            embedding_dim (int): Size of embedding dimension to use on the aforementioned word embeddings
            hidden_dim (int): Size of hidden vectors in LSTM layers
            output_dim (int): Size of output vector from MLP layer

        Kwargs:
            charlm (bool): Whether or not to use the charlm embeddings
            charlm_forward_file (str): The path to the forward pass model for the character language model
            charlm_backward_file (str): The path to the forward pass model for the character language model.
        
        Raises:
            FileNotFoundError: if the forward or backward charlm file cannot be found.
        """
        super(LemmaClassifierLSTM, self).__init__()
        self.input_size = 0
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)
    
        self.embedding_dim = embedding_dim
        self.input_size += embedding_dim

        self.embedding = pt_embedding
        self.vocab_map = vocab_map

        # TODO: pass this up to the parent class
        self.label_decoder = label_decoder

        # Optionally, include charlm embeddings  
        self.use_charlm = kwargs.get("charlm")

        if self.use_charlm:
            charlm_forward_file = kwargs.get("charlm_forward_file")
            charlm_backward_file = kwargs.get("charlm_backward_file")
            if charlm_forward_file is None or not os.path.exists(charlm_forward_file):
                raise FileNotFoundError(f'Could not find forward character model: {kwargs.get("charlm_forward_file", "FILE_NOT_PROVIDED")}')
            if charlm_backward_file is None or not os.path.exists(charlm_backward_file):
                raise FileNotFoundError(f'Could not find backward character model: {kwargs.get("charlm_backward_file", "FILE_NOT_PROVIDED")}')
            add_unsaved_module('charmodel_forward', CharacterLanguageModel.load(charlm_forward_file, finetune=False))
            add_unsaved_module('charmodel_backward', CharacterLanguageModel.load(charlm_backward_file, finetune=False))
            
            self.input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
        
        self.lstm = nn.LSTM(
            self.input_size, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
                        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, pos_indices: List[int], sentences: List[List[str]]):
        """
        Computes the forward pass of the neural net

        Args:
            pos_indices (List[int]): A list of the position index of the target token for lemmatization classification in each sentence.
            sentences (List[List[str]]): A list of the token-split sentences of the input data.

        Returns:
            torch.tensor: Output logits of the neural network, where the shape is  (n, output_size) where n is the number of sentences.
        """
        token_ids = []
        for words in sentences:
            sentence_token_ids = [self.vocab_map.get(word.lower(), UNK_ID) for word in words]
            sentence_token_ids = torch.tensor(sentence_token_ids, device=next(self.parameters()).device)
            token_ids.append(sentence_token_ids)
        
        embedded = self.embedding(torch.tensor(token_ids))

        if self.use_charlm:
            char_reps_forward = self.charmodel_forward.build_char_representation(sentences)  # takes [[str]]
            char_reps_backward = self.charmodel_backward.build_char_representation(sentences)

            embedded = torch.cat((embedded, char_reps_forward, char_reps_backward), 1)  

        print(f"Embedding shape: {embedded.shape}. Should be size (batch_size, T, input_size)")   # Should be size (batch_size, T, input_size)
        padded_sequences = pad_sequence(embedded, batch_first=True)
        lengths = torch.tensor([len(seq) for seq in embedded])

        packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True)

        print(f"Packed Sequences shape: {packed_sequences.shape}. Should be size (batch_size, input_size)")  # should be size (batch_size, input_size)
            
        lstm_out, (hidden, _) = self.lstm(packed_sequences)

        # Extract the hidden state at the index of the token to classify
        unpacked_lstm_outputs, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = unpacked_lstm_outputs[torch.arange(unpacked_lstm_outputs.size(0)), pos_indices]

        print(f"LSTM OUT Shape: {lstm_out.shape}. Should be size (batch_size, input_size)")  # Should be size (batch_size, input_size)

        # MLP forward pass
        output = self.mlp(lstm_out)
        print(f"Output shape: {output.shape}. Should be size (batch_size, output_size)")   # should be size (batch_size, output_size)
        return output

    def model_type(self):
        return ModelType.LSTM
