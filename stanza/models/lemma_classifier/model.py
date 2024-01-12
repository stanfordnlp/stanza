import torch
import torch.nn as nn
import os
import logging
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from typing import List, Tuple

from stanza.models.common.vocab import UNK_ID
from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.constants import ModelType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            upos_to_id (Mapping[str, int]): A dictionary mapping UPOS tag strings to their respective IDs
            upos_emb_dim (int): The size of the UPOS tag embeddings 
            num_heads (int): The number of heads to use for attention. If there are more than 0 heads, attention will be used instead of the LSTM.
        
        Raises:
            FileNotFoundError: if the forward or backward charlm file cannot be found.
        """
        super(LemmaClassifierLSTM, self).__init__()
        self.input_size = 0
        self.num_heads = kwargs.get('num_heads', 0)
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
        
        self.upos_emb_dim = kwargs.get("upos_emb_dim", 0)
        self.upos_to_id = kwargs.get("upos_to_id", None)
        if self.upos_emb_dim > 0 and self.upos_to_id is not None:
            self.upos_emb = nn.Embedding(num_embeddings=len(self.upos_to_id), 
                                        embedding_dim=self.upos_emb_dim, 
                                         padding_idx=0)  
            self.input_size += self.upos_emb_dim

        device = next(self.parameters()).device
        # Determine if attn or LSTM should be used
        if self.num_heads > 0:
            self.input_size = utils.round_up_to_multiple(self.input_size, self.num_heads)
            self.multihead_attn = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=self.num_heads, batch_first=True).to(device)
            logging.info(f"Using attention mechanism with embed dim {self.input_size} and {self.num_heads} attention heads.")
        else:
            self.lstm = nn.LSTM(
            self.input_size, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
                        )
            logging.info(f"Using LSTM mechanism.")

        mlp_input_size = hidden_dim * 2 if self.num_heads == 0 else self.input_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, pos_indices: List[int], sentences: List[List[str]], upos_tags: List[List[int]]):  
        """
        Computes the forward pass of the neural net

        Args:
            pos_indices (List[int]): A list of the position index of the target token for lemmatization classification in each sentence.
            sentences (List[List[str]]): A list of the token-split sentences of the input data.
            upos_tags (List[List[int]]): A list of the upos tags for each token in every sentence.

        Returns:
            torch.tensor: Output logits of the neural network, where the shape is  (n, output_size) where n is the number of sentences.
        """
        device = next(self.parameters()).device
        batch_size = len(sentences)
        token_ids = []
        for words in sentences:
            sentence_token_ids = [self.vocab_map.get(word.lower(), UNK_ID) for word in words]
            sentence_token_ids = torch.tensor(sentence_token_ids, device=device)
            token_ids.append(sentence_token_ids)

        token_ids = pad_sequence(token_ids, batch_first=True)
        embedded = self.embedding(token_ids)
        if self.upos_emb_dim > 0:
            upos_tags = [torch.tensor(sentence_tags) for sentence_tags in upos_tags]  # convert internal lists to tensors
            upos_tags = pad_sequence(upos_tags, batch_first=True, padding_value=0).to(device)   
            pos_emb = self.upos_emb(upos_tags)
            embedded = torch.cat((embedded, pos_emb), 2).to(device)
            
        if self.use_charlm:
            char_reps_forward = self.charmodel_forward.build_char_representation(sentences)  # takes [[str]]
            char_reps_backward = self.charmodel_backward.build_char_representation(sentences)

            char_reps_forward = pad_sequence(char_reps_forward, batch_first=True)
            char_reps_backward = pad_sequence(char_reps_backward, batch_first=True)
            
            embedded = torch.cat((embedded, char_reps_forward, char_reps_backward), 2)  

        padded_sequences = pad_sequence(embedded, batch_first=True)
        lengths = torch.tensor([len(seq) for seq in embedded])        
        
        if self.num_heads > 0:
            target_seq_length, src_seq_length = padded_sequences.size(1), padded_sequences.size(1)
            attn_mask = torch.triu(torch.ones(batch_size * self.num_heads, target_seq_length, src_seq_length, dtype=torch.bool), diagonal=1)

            attn_mask = attn_mask.view(batch_size, self.num_heads, target_seq_length, src_seq_length)
            attn_mask = attn_mask.repeat(1, 1, 1, 1).view(batch_size * self.num_heads, target_seq_length, src_seq_length).to(device)
            
            attn_output, attn_weights = self.multihead_attn(padded_sequences, padded_sequences, padded_sequences, attn_mask=attn_mask)
            # Extract the hidden state at the index of the token to classify
            token_reps = attn_output[torch.arange(attn_output.size(0)), pos_indices]

        else:
            packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True)
            lstm_out, (hidden, _) = self.lstm(packed_sequences)
            # Extract the hidden state at the index of the token to classify
            unpacked_lstm_outputs, _ = pad_packed_sequence(lstm_out, batch_first=True)
            token_reps = unpacked_lstm_outputs[torch.arange(unpacked_lstm_outputs.size(0)), pos_indices]

        # MLP forward pass
        output = self.mlp(token_reps)
        return output

    def model_type(self):
        return ModelType.LSTM
