import torch
import json
import os
import random
import numpy as np

class BertNER:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = {y: x for x, y in self.schema.items()}
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    def load_vocab(self, vocab_path):
        with open(vocab_path, encoding="utf8") as f:
            vocab = json.load(f)
        self.config["vocab_size"] = len(vocab)
        return vocab

    def predict(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            res = torch.argmax(res, dim=-1)
        labeled_sentence = ""
        for char, label_index in zip(sentence, res):
            labeled_sentence += char + self.index_to_sign[int(label_index)]
        return labeled_sentence

def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载Bert NER模型
    ner_model = BertNER(config, "model_output/epoch_10.pth")

    # 测试句子
    sentences = [
        "客厅的颜色比较稳重但不沉重相反很好的表现了欧式的感觉给人高雅的味道",
        "双子座的健康运势也呈上升的趋势但下半月有所回落"
    ]

    for sentence in sentences:
        res = ner_model.predict(sentence)
        print(res)

if __name__ == "__main__":
    main(Config)
