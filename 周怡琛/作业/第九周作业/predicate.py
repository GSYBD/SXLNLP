"""
模型效果测试
"""
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel


class Predictor:
    def __init__(self, config, model_path,sentence):
        self.config = config
        self.vocab = self.load_vocab(config["vocab_path"])
        self.data = self.encode_sentence(sentence,padding=True)
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def encode_sentence(self, sentence, padding=True):
        input_id = [4700]
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
            # input_id.append(-2)
        if padding:
            input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=4701):
        input_id = input_id[:self.config["max_length"] - 1]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id) - 1)
        return input_id

    def predict(self,data,sentence):
        with torch.no_grad():
            result = self.model(data)
        result = torch.argmax(result, dim=-1)
        label = result.cpu().detach().tolist()
        entities = self.decode(sentence, label)
        print("=+++++++++")
        print(entities)


    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s-1:e-1])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s-1:e-1])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s-1:e-1])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s-1:e-1])
        return results