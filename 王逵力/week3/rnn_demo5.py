import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch.nn.utils.rnn as rnn_utils

data = [
    ("I love machine learning", 0),
    ("Artificial intelligence is amazing", 1),
    ("Natural language processing", 1),
    ("Deep learning is fun", 1),
    ("RNN can be used for this task", 1),
    ("This sentence has no a", 0),
    ("Neither does this one", 0)
]

def generate_label(sentence):
    return 1 if 'a' in sentence else 0

new_labels = [generate_label(sentence) for sentence, _ in data]

def simple_tokenize(sentences):
    return [sentence.split() for sentence in sentences]

tokenized_data = simple_tokenize([s[0] for s in data])

# Create vocabulary dictionary
vocab = defaultdict(lambda: 0)
for sentence in tokenized_data:
    for word in sentence:
        vocab[word] += 1

# Encode words to indices
word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
encoded_data = [[word2idx[word] for word in sentence] for sentence in tokenized_data]

# Pad sequences to make them the same length
max_len = max(len(sentence) for sentence in encoded_data)
padded_data = [sentence + [0] * (max_len - len(sentence)) for sentence in encoded_data]

# Convert to tensors
padded_data = torch.tensor(padded_data)
encoded_labels = torch.tensor([s[1] for s in data])

# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    padded_data, encoded_labels, test_size=0.25, random_state=42
)

# Custom collate function for DataLoader
def collate_fn(batch):
    inputs, targets = zip(*batch)
    # Pad sequences
    inputs = rnn_utils.pad_sequence(inputs, batch_first=True)
    targets = torch.tensor(targets)
    return inputs, targets

# Dataset class
class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create datasets and data loaders
train_dataset = TextDataset(train_data, train_labels)
test_dataset = TextDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Define RNN model
class TextClassifierRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextClassifierRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is now (batch_size, seq_len)
        embedded = self.embedding(x)
        lengths = [len(seq) for seq in x]
        packed = rnn_utils.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        # Unpack the sequences
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        # Use the last non-padded element of the last sequence as the output
        idx = (torch.tensor(lengths) - 1).view(-1, 1).expand(x.size(0), output.size(2))
        hidden = output.gather(1, idx.unsqueeze(1)).squeeze(1)
        return self.fc(hidden)

# Model parameters
vocab_size = len(vocab) + 1  # Including padding index
embed_dim = 50  # Embedding dimension
hidden_dim = 100  # RNN hidden layer dimension
num_classes = 2  # Number of classes

model = TextClassifierRNN(vocab_size, embed_dim, hidden_dim, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
    return test_loss / len(test_loader)

# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
test_loss = evaluate_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss}')