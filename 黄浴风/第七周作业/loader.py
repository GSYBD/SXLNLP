import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.label_to_index = {'0': 0, '1': 1}  # 将标签映射到 0 或 1
        self.index_to_label = {0: '0', 1: '1'}
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    # def load(self):
    #     self.data = []
    #     with open(self.path, encoding="utf8") as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             label = row[0]  # Assuming label is in the first column
    #             text = row[1]  # Assuming text is in the second column
    #             label = self.label_to_index[label]
    #             if self.config["model_type"] == "bert":
    #                 input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], pad_to_max_length=True)
    #             else:
    #                 input_id = self.encode_sentence(text)
    #             input_id = torch.LongTensor(input_id)
    #             label_index = torch.LongTensor([label])
    #             self.data.append([input_id, label_index])
    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            reader = csv.reader(f)
            for row in reader:
                label = row[0]  # Assuming label is in the first column
                text = row[1]  # Assuming text is in the second column
                
                # Debugging statements to check the actual label value
                
                
                # Check if label exists in label_to_index dictionary
                if label in self.label_to_index:
                    label = self.label_to_index[label]
                else:
                    print("Label not found in label_to_index dictionary.")
                    continue
                
                # Continue with data processing
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(text)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = [self.vocab.get(char, self.vocab["[UNK]"]) for char in text]
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
    
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

# 修改标签为0或1的数据加载函数
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("val_data.csv", Config)  # 用你的 CSV 文件路径替换 "your_csv_file.csv"
    print(dg[1])
