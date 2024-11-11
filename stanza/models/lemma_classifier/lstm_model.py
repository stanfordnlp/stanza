import torch
import torch.nn as nn
import os
import logging
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from typing import List, Tuple

from stanza.models.common.vocab import UNK_ID
from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.constants import ModelType

logger = logging.getLogger('stanza.lemmaclassifier')

class LemmaClassifierLSTM(LemmaClassifier):
    """
    Model architecture:
        Extracts word embeddings over the sentence, passes embeddings into a bi-LSTM to get a sentence encoding.
        From the LSTM output, we get the embedding of the specific token that we classify on. That embedding
        is fed into an MLP for classification.
    """
    def __init__(self, model_args, output_dim, pt_embedding, label_decoder, upos_to_id, known_words, target_words, target_upos,
                 use_charlm=False, charlm_forward_file=None, charlm_backward_file=None):
        """
        Args:
            vocab_size (int): Size of the vocab being used (if custom vocab)
            output_dim (int): Size of output vector from MLP layer
            upos_to_id (Mapping[str, int]): A dictionary mapping UPOS tag strings to their respective IDs
            pt_embedding (Pretrain): pretrained embeddings
            known_words (list(str)): Words which are in the training data
            target_words (set(str)): a set of the words which might need lemmatization
            use_charlm (bool): Whether or not to use the charlm embeddings
            charlm_forward_file (str): The path to the forward pass model for the character language model
            charlm_backward_file (str): The path to the forward pass model for the character language model.

        Kwargs:
            upos_emb_dim (int): The size of the UPOS tag embeddings
            num_heads (int): The number of heads to use for attention. If there are more than 0 heads, attention will be used instead of the LSTM.

        Raises:
            FileNotFoundError: if the forward or backward charlm file cannot be found.
        """
        super(LemmaClassifierLSTM, self).__init__(label_decoder, target_words, target_upos)
        self.model_args = model_args

        self.hidden_dim = model_args['hidden_dim']
        self.input_size = 0
        self.num_heads = self.model_args['num_heads']

        emb_matrix = pt_embedding.emb
        self.add_unsaved_module("embeddings", nn.Embedding.from_pretrained(emb_matrix, freeze=True))
        self.vocab_map = { word.replace('\xa0', ' '): i for i, word in enumerate(pt_embedding.vocab) }
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        self.known_words = known_words
        self.known_word_map = {word: idx for idx, word in enumerate(known_words)}
        self.delta_embedding = nn.Embedding(num_embeddings=len(known_words)+1,
                                            embedding_dim=self.embedding_dim,
                                            padding_idx=0)
        nn.init.normal_(self.delta_embedding.weight, std=0.01)

        self.input_size += self.embedding_dim

        # Optionally, include charlm embeddings
        self.use_charlm = use_charlm

        if self.use_charlm:
            if charlm_forward_file is None or not os.path.exists(charlm_forward_file):
                raise FileNotFoundError(f'Could not find forward character model: {charlm_forward_file}')
            if charlm_backward_file is None or not os.path.exists(charlm_backward_file):
                raise FileNotFoundError(f'Could not find backward character model: {charlm_backward_file}')
            self.add_unsaved_module('charmodel_forward', CharacterLanguageModel.load(charlm_forward_file, finetune=False))
            self.add_unsaved_module('charmodel_backward', CharacterLanguageModel.load(charlm_backward_file, finetune=False))

            self.input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()

        self.upos_emb_dim = self.model_args["upos_emb_dim"]
        self.upos_to_id = upos_to_id
        if self.upos_emb_dim > 0 and self.upos_to_id is not None:
            # TODO: should leave space for unknown POS?
            self.upos_emb = nn.Embedding(num_embeddings=len(self.upos_to_id),
                                         embedding_dim=self.upos_emb_dim,
                                         padding_idx=0)
            self.input_size += self.upos_emb_dim

        device = next(self.parameters()).device
        # Determine if attn or LSTM should be used
        if self.num_heads > 0:
            self.input_size = utils.round_up_to_multiple(self.input_size, self.num_heads)
            self.multihead_attn = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=self.num_heads, batch_first=True).to(device)
            logger.debug(f"Using attention mechanism with embed dim {self.input_size} and {self.num_heads} attention heads.")
        else:
            self.lstm = nn.LSTM(self.input_size,
                                self.hidden_dim,
                                batch_first=True,
                                bidirectional=True)
            logger.debug(f"Using LSTM mechanism.")

        mlp_input_size = self.hidden_dim * 2 if self.num_heads == 0 else self.input_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def get_save_dict(self):
        save_dict = {
            "params": self.state_dict(),
            "label_decoder": self.label_decoder,
            "model_type": self.model_type().name,
            "args": self.model_args,
            "upos_to_id": self.upos_to_id,
            "known_words": self.known_words,
            "target_words": list(self.target_words),
            "target_upos": list(self.target_upos),
        }
        skipped = [k for k in save_dict["params"].keys() if self.is_unsaved_module(k)]
        for k in skipped:
            del save_dict["params"][k]
        return save_dict

    def convert_tags(self, upos_tags: List[List[str]]):
        if self.upos_to_id is not None:
            return [[self.upos_to_id[x] for x in sentence] for sentence in upos_tags]
        return None

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
        delta_token_ids = []
        for words in sentences:
            sentence_token_ids = [self.vocab_map.get(word.lower(), UNK_ID) for word in words]
            sentence_token_ids = torch.tensor(sentence_token_ids, device=device)
            token_ids.append(sentence_token_ids)

            sentence_delta_token_ids = [self.known_word_map.get(word.lower(), 0) for word in words]
            sentence_delta_token_ids = torch.tensor(sentence_delta_token_ids, device=device)
            delta_token_ids.append(sentence_delta_token_ids)

        token_ids = pad_sequence(token_ids, batch_first=True)
        delta_token_ids = pad_sequence(delta_token_ids, batch_first=True)
        embedded = self.embeddings(token_ids) + self.delta_embedding(delta_token_ids)

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

        if self.num_heads > 0:

            def positional_encoding(seq_len, d_model, device):
                encoding = torch.zeros(seq_len, d_model, device=device)
                position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)

                encoding[:, 0::2] = torch.sin(position * div_term)
                encoding[:, 1::2] = torch.cos(position * div_term)

                # Add a new dimension to fit the batch size
                encoding = encoding.unsqueeze(0)
                return encoding

            seq_len, d_model = embedded.shape[1], embedded.shape[2]
            pos_enc = positional_encoding(seq_len, d_model, device=device)

            embedded += pos_enc.expand_as(embedded)

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
