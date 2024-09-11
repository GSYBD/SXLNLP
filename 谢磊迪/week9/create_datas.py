import json
from importlib import reload
import torch
from torch.utils.data import DataLoader
import config
from transformers import BertTokenizer
reload(config)
config = config.config


class Create_Datas:
    def __init__(self, config, train_or_test_path):
        self.config = config
        self.vocab_path = config["vocab_path"]
        self.train_or_test_path = config[train_or_test_path]
        self.sent_len = config["sent_len"]
        self.bert_path = config['bert_path']
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.schema_path = config["schema_path"]
        # self.vocab2int()
        self.schema_label()
        self.create_datas_first()
        print("create_datas_first ok ")
        self.create_datas_last()
        print("create_datas_last ok")

    def vocab2int(self):

        self.vocab_dic = {}
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            for ind, word in enumerate(f.readlines()):
                self.vocab_dic[word.strip()] = ind + 1
            # [vocab_dic.get(word, "[UNK]") for word in word_list]
        config["vocab_size"] = len(self.vocab_dic) + 1
        print("vocab2int ok")

    def schema_label(self):
        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema_dic = json.load(f)
        print("schema ok")

    # def padding(self, input_id, pad_token=0):
    #     input_id = input_id[: self.config["max_length"]]
    #     input_id += [pad_token] * (self.config["max_length"] - len(input_id))
    #     return input_id

    def padding1(self,label_list):
        # _word_list = word_list[: self.config["sent_len"]]
        # _word_list += [0] * (self.config["sent_len"] - len(_word_list))
        _label_list = label_list[: self.config["sent_len"]]
        _label_list += [-1] * (self.config["sent_len"] - len(_label_list))
        # _word_list,
        return _label_list

    def create_datas_first(self):
        with open(self.train_or_test_path, "r", encoding="utf-8") as f:
            self.word_file = []
            self.label_file = []
            for i in f.read().strip().split("\n\n"):
                word_line = []
                label_line = ["O"]
                for word_label in i.split("\n"):
                    word, label = word_label.split(" ")
                    word_line.append(word)
                    label_line.append(label)
                self.word_file.append(word_line)
                self.label_file.append(label_line)
            self.sentences = ["".join(i) for i in self.word_file]

    def create_datas_last(self):
        self.data_finall = []
        for sentence, label_list in zip(self.sentences, self.label_file):
            _label_list = self.padding1(label_list)
            _word_list  = torch.LongTensor(self.tokenizer.encode(sentence,padding='max_length',max_length=self.sent_len,truncation=True))
            # _word_list = torch.LongTensor([self.vocab_dic.get(word, self.vocab_dic["[UNK]"]) for word in _word_list])
            _label_list = torch.LongTensor([self.schema_dic.get(word, -1) for word in _label_list])
            self.data_finall.append([_word_list, _label_list])

    def __len__(self):
        return len(self.data_finall)

    def __getitem__(self, idx):
        return self.data_finall[idx]


def load_datas(config, train_or_test_path):
    dataset = Create_Datas(config, train_or_test_path)
    dl = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    return dl


if __name__ == "__main__":
    train_or_test_path = "test_path"
    dataset = Create_Datas(config, train_or_test_path)
    print(dataset.sentences)
    # dl = load_datas(config,train_or_test_path)
    # print(config["vocab_size"])
