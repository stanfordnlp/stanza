import torch
import torch.nn as nn
import torch.optim as optim
import os
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer
from transformers import BertTokenizer, BertModel


class LemmaClassifierWithTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embeddings, padding_idx = 0, **kwargs):
        super(LemmaClassifierWithTransformer, self).__init__()

        # Get the embedding through transformer 

        # define an MLP layerr
    
    def forward(self):

        # Get the transformer embeddings 

        # pass to the MLP
        pass 


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample sentence
sentence = "Contextual embeddings with BERT."

# Tokenize and prepare input
tokens = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# Forward pass through BERT model
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

# Extract embeddings from a specific layer (e.g., last layer)
last_hidden_states = outputs.last_hidden_state

# Assuming you want the embeddings for the entire sequence
word_embeddings = last_hidden_states[0]
print(word_embeddings, word_embeddings.shape)
# Tokenize the word "Contextual"
tokenized_word = tokenizer.tokenize("Contextual")

# Find the position of the tokenized word in the list of tokens
word_index = tokens['input_ids'][0].tolist().index(tokenizer.convert_tokens_to_ids(tokenized_word[0]))

# Extract the corresponding embedding from word_embeddings
contextual_embedding_for_word = last_hidden_states[0, word_index]
print(contextual_embedding_for_word, contextual_embedding_for_word.shape)

# Now, contextual_embedding_for_word contains the representation for the word "Contextual"





