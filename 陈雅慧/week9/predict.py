# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
from config import Config
from transformers import BertModel
from transformers import BertTokenizer
from evaluate import Evaluator
from loader import DataGenerator
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
def predict(sentence,model):
    model.load_state_dict(torch.load(r"C:\Users\CYH\Desktop\学习\课件\week9 序列标注\week9 序列标注问题\作业\ner\model_output\epoch_15.pth"))
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
