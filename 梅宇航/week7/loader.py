# loader.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=30):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_id = self.encode_sentence(text)
        return torch.tensor(input_id, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def encode_sentence(self, text):
        words = list(text.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace(',', ' ').replace('!', ' ').split())
        input_id = [self.vocab.get(word, self.vocab["<PAD>"]) for word in words]
        input_id = input_id[:self.max_length] + [self.vocab["<PAD>"]] * (self.max_length - len(input_id))
        return input_id

def load_data(file_path, vocab, batch_size=32, max_length=30, shuffle=True):
    data = pd.read_csv(file_path, header=None, names=['label', 'text'])
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    dataset = TextDataset(texts, labels, vocab, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def create_vocab(data, vocab_size=5000):
    all_words = []
    for text in data['text']:
        words = list(text.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace(',', ' ').replace('!', ' ').split())
        all_words.extend(words)
    word_counter = Counter(all_words)
    most_common_words = word_counter.most_common(vocab_size - 1)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common_words)}
    vocab["<PAD>"] = 0
    return vocab

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab
