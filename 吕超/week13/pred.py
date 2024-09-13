import torch
import logging
from model import TorchModel
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

from evaluate import Evaluator
from config import Config


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#大模型微调策略
tuning_tactics = Config["tuning_tactics"]

print("正在使用 %s"%tuning_tactics)

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

#重建模型
model = TorchModel(Config)
# print(model.state_dict().keys())
# print("====================")

model = get_peft_model(model, peft_config)
# print(model.state_dict().keys())
# print("====================")

state_dict = model.state_dict()

#将微调部分权重加载
if tuning_tactics == "lora_tuning":
    loaded_weight = torch.load('model_output/lora_tuning.pth')
elif tuning_tactics == "p_tuning":
    loaded_weight = torch.load('model_output/p_tuning.pth')
elif tuning_tactics == "prompt_tuning":
    loaded_weight = torch.load('model_output/prompt_tuning.pth')
elif tuning_tactics == "prefix_tuning":
    loaded_weight = torch.load('model_output/prefix_tuning.pth')

print(loaded_weight.keys())
state_dict.update(loaded_weight)

#权重更新后重新加载到模型
model.load_state_dict(state_dict)

#进行一次测试
# model = model.cuda()
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)


# # -*- coding: utf-8 -*-
# import torch
# import re
# import json
# import numpy as np
# from collections import defaultdict
# from config import Config
# from model import TorchModel
# from transformers import BertTokenizer, BertModel
# from peft import get_peft_model, LoraConfig
# # from main import peft_wrapper
#
# """
# 模型效果测试
# """
#
#
# class NER:
#     def __init__(self, config, model_path):
#         self.config = config
#         self.tokenizer = self.load_vocab(config["bert_path"])
#         self.schema = self.load_schema(config["schema_path"])
#         model = TorchModel(config)
#         # model = peft_wrapper(model)
#         model = get_peft_model(model, peft_config)
#         state_dict = model.state_dict()
#         state_dict.update(torch.load(model_path))
#         model.load_state_dict(state_dict)
#         model.eval()
#         self.model = model
#         print("模型加载完毕!")
#
#     def load_schema(self, path):
#         with open(path, encoding="utf8") as f:
#             return json.load(f)
#
#     # 加载字表或词表
#     def load_vocab(self, vocab_path):
#         return BertTokenizer.from_pretrained(vocab_path)
#
#     def encode_sentence(self, text, padding=True):
#         return self.tokenizer.encode(text,
#                                      padding="max_length",
#                                      max_length=self.config["max_length"],
#                                      truncation=True)
#
#     def decode(self, sentence, labels):
#         sentence = "$" + sentence
#         labels = "".join([str(int(x)) for x in labels[:len(sentence) + 1]])
#         results = defaultdict(list)
#
#         for location in re.finditer("(04+)", labels):
#             s, e = location.span()
#             results["LOCATION"].append(sentence[s:e])
#         for location in re.finditer("(15+)", labels):
#             s, e = location.span()
#             results["ORGANIZATION"].append(sentence[s:e])
#         for location in re.finditer("(26+)", labels):
#             s, e = location.span()
#             results["PERSON"].append(sentence[s:e])
#         for location in re.finditer("(37+)", labels):
#             s, e = location.span()
#             results["TIME"].append(sentence[s:e])
#         return results
#
#     def predict(self, sentence):
#         input_ids = self.encode_sentence(sentence)
#         with torch.no_grad():
#             res = self.model(torch.LongTensor([input_ids]))[0]
#             labels = torch.argmax(res, dim=-1)
#         results = self.decode(sentence, labels)
#         return results
#
#
# if __name__ == "__main__":
#     sl = NER(Config, "model_output/lora_tuning.pth")
#     sentence = "(本报约翰内斯堡电)本报记者安洋贺广华留学人员档案库建立本报讯中国质量体系认证机构国家认可委员会日前正式签署了国际上第一个质量认证的多边互认协议,表明中国质量体系认证达到了国际水平。"
#     res = sl.predict(sentence)
#     print(res)
