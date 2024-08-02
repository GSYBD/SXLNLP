import pandas as pd
import torch
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        self.load()

    def load(self):
        self.data = []
        df = pd.read_csv(self.data_path)
        data_label_0 = df[df["label"] == 0]
        data_label_1 = df[df["label"] == 1]
        for i in range(len(data_label_0)):
            self.data.append(
                [torch.LongTensor(encode_sentence(data_label_0["review"].iloc[i], self.vocab, self.config)),
                 torch.LongTensor([0])])
        for i in range(len(data_label_1)):
            self.data.append(
                [torch.LongTensor(encode_sentence(data_label_1["review"].iloc[i], self.vocab, self.config)),
                 torch.LongTensor([1])])
            self.data.append(
                [torch.LongTensor(encode_sentence(data_label_1["review"].iloc[i], self.vocab, self.config)),
                 torch.LongTensor([1])])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            vocab[line.strip()] = index + 1  # 留出pudding 0
    return vocab


def encode_sentence(sentence, vocab, config):
    input_sequence = []
    for char in sentence:
        input_sequence.append(vocab.get(char,vocab["[UNK]"]))
    input_sequence = padding(input_sequence, config)
    return input_sequence


def padding(input_sequence, config):
    input_sequence = input_sequence[:config["max_length"]]
    input_sequence += [0] * (config["max_length"] - len(input_sequence))
    return input_sequence


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

# if __name__=="__main__":
#     from config import Config
#     dg = DataGenerator(Config["train_data_path"], Config)
#     print(len(dg.vocab))
#     # print(dg[1])
#     dl=load_data(Config["train_data_path"],Config)
#     cuda_flag = torch.cuda.is_available()
#     for index,batch_data in enumerate(dl):
#         input_ids, labels = batch_data
#         print(input_ids.shape)
#         print(labels.shape)
