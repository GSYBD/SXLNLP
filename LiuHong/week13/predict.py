# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from model import TorchModel
from config import Config
from transformers import BertModel
from transformers import BertTokenizer
import logging
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
模型效果测试
"""
'''
{
  "B-LOCATION": 0,
  "B-ORGANIZATION": 1,
  "B-PERSON": 2,
  "B-TIME": 3,
  "I-LOCATION": 4,
  "I-ORGANIZATION": 5,
  "I-PERSON": 6,
  "I-TIME": 7,
  "O": 8
}
'''
def predict(sentence):
    # 大模型微调策略
    tuning_tactics = Config["tuning_tactics"]

    print("正在使用 %s" % tuning_tactics)

    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["embedding", "query", "key", "value"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    # 重建模型
    model = TorchModel(Config)
    # print(model.state_dict().keys())
    # print("====================")

    model = get_peft_model(model, peft_config)
    # print(model.state_dict().keys())
    # print("====================")

    state_dict = model.state_dict()
    # 将微调部分权重加载
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

    # 权重更新后重新加载到模型
    model.load_state_dict(state_dict)
    model.eval()   #测试模式，不使用dropout
    x=BertTokenizer.from_pretrained(Config["bert_path"]).encode(sentence)
    x=torch.LongTensor(x)
    x=x.unsqueeze(0)
    with torch.no_grad():  #不计算梯度
        result = model(x)  #模型预测
    result=torch.argmax(result,dim=2)
    result=result.tolist()[0]

    labels = "".join([str(x) for x in result[:len(sentence)]])
    results = defaultdict(list)
    for location in re.finditer("(04+)", labels):
        s, e = location.span()
        results["LOCATION"].append(sentence[s-1:e-1])#增加的cls的原因，要从s-1开始取
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