import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertModel

"""
目标：
实现bert的这个算法。只算一层
直接拿已经练好的参数（模型），然后做预测，只预测输出第一层，看结果是否一致

要做的：
1.实现加载bert并只预测第一层

2.按理解拿已有的参数实现计算过程

3.做结果对比

看着不多hh
"""

bert = BertModel.from_pretrained('')

