import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer

# Define a custom model for your binary classifier
class LemmaClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embeddings):
        super(LemmaClassifier, self).__init__()

        # Embedding layer with GloVe embeddings
        self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),  # Add 1 for positional index
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, text, pos_index):
        # Token embeddings
        embedded = self.embedding(text)

        # LSTM forward pass
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Extract the last hidden state
        print(lstm_out)
        lstm_out = lstm_out[-1]

        # Concatenate the last hidden state with positional index
        print(f"LSTM OUT: {lstm_out}, shape: {lstm_out.shape}")
        print(f"POS INDEX: {pos_index}, shape: {pos_index.shape}")
        # concatenated = torch.cat((lstm_out, pos_index), dim=1)
        concatenated = torch.cat((lstm_out, pos_index), dim=0)
        print(concatenated, concatenated.shape)


        # MLP forward pass
        output = self.mlp(concatenated)

        print(output)

        return output

# Define the hyperparameters
vocab_size = 10000  # Adjust based on your dataset
embedding_dim = 100
hidden_dim = 256
output_dim = 2  # Binary classification (be or have)

# Load GloVe embeddings (adjust the path and dimensions)
glove = GloVe(name='6B', dim=embedding_dim)
vocab_size = len(glove.itos)  # Size of the GloVe vocabulary

# Initialize the model with GloVe embeddings
model = LemmaClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, glove.vectors)

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample input data (make sure to adapt this to your dataset)
tokenizer = get_tokenizer('basic_english')
sentence = "The cat's tail is long"
# tokenized_sentence = tokenizer(sentence)  # Tokenize the input sentence  , maybe use stanza next time
tokenized_sentence = ['the', 'cat', "'s", 'tail', 'is', 'long']

# Convert the tokenized input to a tensor
# tokenized_text = [word for word in tokenized_sentence]
positional_index = tokenized_sentence.index("'s")
tokenized_text = torch.tensor([glove.stoi[word] for word in tokenized_sentence])   # maybe just word not glove.stoi[word]

print(tokenized_text)

positional_index = torch.tensor([positional_index])  # Convert positional index to tensor

# Forward pass
output = model(tokenized_text, positional_index)
print(output)
# output_probs = torch.softmax(output, dim=0)

# # Choose the class with the highest probability as the prediction
# predicted_class = torch.argmax(output_probs, dim=0)

# print(predicted_class, predicted_class.shape)

# Compute the loss
# target = torch.tensor([0], dtype=torch.long)  # 0 for "be" and 1 for "have"
# Define the target label (0 for "be" and 1 for "have")
target = torch.tensor([0, 1], dtype=torch.float)  # Use dtype=torch.long for class labels, but torch.float seems to make this work
print(target)
loss = criterion(output, target)
print(loss)

# Backpropagation and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Training loop and evaluation code can be added here
