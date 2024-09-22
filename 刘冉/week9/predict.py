# coding: utf-8

import torch
import json
import os
from config import Config
from model import PunctuationModel
from transformers import BertTokenizer

class Predictor:
    def __init__(self, config):
        self.config = config
        self.max_length = config["max_length"]
        self.schema = self.load_schema(config["schema_path"])
        config["class_num"] = len(self.schema)
        self.model = PunctuationModel(config)
        model_path = os.path.join(config["model_path"], "punctuation_model.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # 确保模型在正确的设备上
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        #token
        self.padding = config["padding"]
        self.bertTokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"], pad_token_id=self.padding)


    def predict(self, sentence):
        self.model.eval()
        sentence = sentence[0:self.max_length]
        encode_sentence = self.bertTokenizer.encode(sentence, max_length=self.max_length, padding="max_length")
        if torch.cuda.is_available():
            encode_sentence = encode_sentence.cuda()
        with torch.no_grad():
           result = self.model(torch.LongTensor([encode_sentence]))
        result = torch.argmax(result, dim=-1)
        result = result.cpu().detach().tolist()[0]
        result = result[1:len(sentence)+1]
        assert len(result) == len(sentence) , print(len(result), len(sentence))
        pred_sen = ""
        for label, char in zip(result, sentence):
            pred_sen += char
            for key, value in self.schema.items():
                if label == value:
                    pred_sen += key
                    break
        return pred_sen

    def load_schema(self, schema_path):
        with open(schema_path, encoding="utf8") as f:
            return json.load(f)

if __name__ == "__main__":

    predictor = Predictor(Config)
    sentence = "客厅的颜色比较稳重但不沉重相反很好的表现了欧式的感觉给人高雅的味道"
    res = predictor.predict(sentence)
    print(res)

    sentence = "双子座的健康运势也呈上升的趋势但下半月有所回落"
    res = predictor.predict(sentence)
    print(res)

    result = predictor.predict("你以为破财都是会在预料之中吗小心一点还是可以避掉应该还可以安全过关")
    print(result)