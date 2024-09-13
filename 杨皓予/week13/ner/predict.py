# -*- coding: utf-8 -*-
import logging

import torch
import re
import json
import numpy as np
from collections import defaultdict
from evaluate import Evaluator
from transformers import BertTokenizer
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

from config import Config
from model import TorchModel

"""
模型效果测试
"""

# logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# 大模型微调策略
tuning_tactics = Config["tuning_tactics"]

print("正在使用 %s" % tuning_tactics)

if tuning_tactics == "lora_tuning":
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
elif tuning_tactics == "p_tuning":
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prompt_tuning":
    peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prefix_tuning":
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
#
# #重建模型
# model = TorchModel(Config)
# # print(model.state_dict().keys())
# # print("====================")
#
# model = get_peft_model(model, peft_config)
# # print(model.state_dict().keys())
# # print("====================")
#
# state_dict = model.state_dict()
#
# 将微调部分权重加载
if tuning_tactics == "lora_tuning":
    loaded_weight = torch.load('model_output/epoch_8.pth')
elif tuning_tactics == "p_tuning":
    loaded_weight = torch.load('output/p_tuning.pth')
elif tuning_tactics == "prompt_tuning":
    loaded_weight = torch.load('output/prompt_tuning.pth')
elif tuning_tactics == "prefix_tuning":
    loaded_weight = torch.load('output/prefix_tuning.pth')


#
# print(loaded_weight.keys())
# state_dict.update(loaded_weight)
#
# #权重更新后重新加载到模型
# model.load_state_dict(state_dict)
#
# #进行一次测试
#
# evaluator = Evaluator(Config, model, logger)
# evaluator.eval(0)


class SentenceLabel:
    def __init__(self, config):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.tokenizer = load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model = get_peft_model(self.model, peft_config)
        self.state_dict = self.model.state_dict()
        self.state_dict.update(loaded_weight)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    def encode_sentence(self, review):
        input_id = self.tokenizer.encode(review, max_length=self.config["max_length"],
                                         truncation=True,
                                         padding="max_length")
        return input_id

    def predict(self, sentence):
        input_id = self.tokenizer.encode(sentence)
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            res = torch.argmax(res, dim=-1)
            print(decode(sentence, res.tolist()))
        #     res = res[1:-1]
        # labeled_sentence = ""
        # for char, label_index in zip(sentence, res):
        #     if self.index_to_sign[int(label_index)] == "O":
        #         labeled_sentence += char
        #         continue
        #     labeled_sentence += char + self.index_to_sign[int(label_index)]
        # return labeled_sentence


def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer


def decode(sentence, labels):
    sentence = "$" + sentence
    labels = "".join([str(x) for x in labels[:len(sentence) + 1]])
    results = defaultdict(list)
    for location in re.finditer("(04+)", labels):
        s, e = location.span()
        results["LOCATION"].append(sentence[s:e])
    for location in re.finditer("(15+)", labels):
        s, e = location.span()
        results["ORGANIZATION"].append(sentence[s:e])
    for location in re.finditer("(26+)", labels):
        s, e = location.span()
        results["PERSON"].append(sentence[s:e])
    for location in re.finditer("(37+)", labels):
        s, e = location.span()
        results["TIME"].append(sentence[s:e])
    return results


if __name__ == "__main__":
    sl = SentenceLabel(Config)

    # sentence = "他说:中国政府对目前南亚出现的核军备竞赛的局势深感忧虑和不安。"
     # 测试句子
    sentence = "我在北京的清华大学读书。"
    res = sl.predict(sentence)
