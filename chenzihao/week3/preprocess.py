import torch


import os
from collections import Counter

def read_file(filename):
    
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            label, content = line.strip().split('\t')
            if content:
                contents.append(content)
                labels.append(label)
    return contents, labels


def build_vocab(train_file, vocab_file, vocab_size = 5000):
    data_train, _ = read_file(train_file)

    data = []
    for content in data_train: # content is string
        data.extend(content) # 自动加字的

    counter = Counter(data)
    count_pairs = counter.most_common(vocab_size - 2)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words) + ['<UNK>'] # check whether we need ['<UNK>']

    with open(vocab_file, 'w', encoding= "utf-8") as f:
        f.write( '\n'.join(words) + '\n')
    
    return dict(zip(words, range(len(words))))

def read_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        words = [ w.rstrip('\n') for w in f.readlines()] # .strip会导致删除了空格
    word_dict = dict(zip(words, range(len(words))))
    return word_dict

def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [x for x in categories]

    cat_dict = dict(zip(categories, range(len(categories))))

    return cat_dict

def id2word(ids, word_dict):
    return ''.join(word_dict[x] for x in ids)

def word2id(input, word_dict, ids): # this is not a generation task no need to inverse
    pass


from torch.utils.data import Dataset

class TCDataset(Dataset):
    def __init__(self, contents, labels, vocab) -> None:
        # super().__init__() # no need to super init
        self.contents = contents
        self.labels = labels
        self.vocab = vocab
        self.category = read_category()

    def __len__(self):
        return len(self.contents)
    
    def __getitem__(self, index):
        content = self.contents[index]
        temp = self.labels[index]
        
        label = self.category[temp]
    
        ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in content]
        
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype = torch.long)

def get_dataset(train_file, val_file, test_file, vocab_file):
    build_vocab(train_file, vocab_file)
    vocab = read_vocab(vocab_file)

    contents, labels = read_file(train_file)
    train = TCDataset(contents, labels, vocab)

    contents, labels = read_file(val_file)
    val = TCDataset(contents, labels, vocab)

    contents, labels = read_file(test_file)
    test = TCDataset(contents, labels, vocab)

    return train, val, test

if __name__ == "__main__":
    contents, labels = read_file('cnews/cnews.train.txt')
    build_vocab('cnews/cnews.train.txt', 'vocab.txt')
    vocab = read_vocab('vocab.txt')
    
    # dataset = TCDataset(contents, labels, vocab)
    # print(dataset[1])

    print(len(vocab))
