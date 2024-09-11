import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# 假设 load_vocab 函数已经定义
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0 留给 padding 位置，所以从 1 开始
    return token_dict

class DataGenerator(Dataset):
    def __init__(self, data, config):
        self.config = config
        self.data = data
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config['class_num']=2
        self.config["vocab_size"] = len(self.vocab)
        self.process_data()

    def process_data(self):
        processed_data = []
        for label, title in self.data:
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], padding='max_length', truncation=True)
            else:
                input_id = self.encode_sentence(title)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            processed_data.append([input_id, label_index])
        self.data = processed_data

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(data_path, config, shuffle=True):
    df = pd.read_csv(data_path, encoding="utf8")
    data = [(int(row.iloc[0]), row.iloc[1]) for _, row in df.iterrows()]
    train_data, test_data = train_test_split(data, test_size=config["test_size"], random_state=config["random_state"], shuffle=True)

    train_dataset = DataGenerator(train_data, config)
    test_dataset = DataGenerator(test_data, config)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    from config import Config
    train_loader, test_loader = load_data("data.csv", Config)
    print(next(iter(train_loader)))
    print(next(iter(test_loader)))