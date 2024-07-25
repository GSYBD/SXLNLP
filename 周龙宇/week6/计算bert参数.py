import torch
import math
import numpy as np
from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"D:\pythonfile\华中杯\八斗AI\week6 语言模型和预训练\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
'''
  ^ 表示 平方 等同于 **
1.词嵌入层：
    v是词汇表大小，h是隐藏层维度
    词嵌入矩阵：  V X H
    位置编码矩阵：  512 X H
    token类型矩阵：  2 X H
    layernorm层：  H + H = 2H
    总参数量：  (V + 512 + 2 + 2) X H  = (V + 516) X H
2.Transformer层
    2.1多头注意力层
        Q矩阵：  H X H + H
        K矩阵：  H X H + H
        V矩阵：  H X H + H
        linear层矩阵：  H X H + H
        总参数量：  4H^2 + 4H

    2.2前馈网络层
        linear层矩阵1：  H X 4H + 4H
        linear层矩阵2：  4H X H + H
        总参数量：  8H^2 + 5H

    2.3残差连接+层归一化
        layernorm层1：  H + H = 2H
        layernorm层2：  H + H = 2H
        总参数量：  4H

    总参数量：  12H^2 + 13 H

3.总transformer组合
    L X Transformer数量层： L X (12H^2 + 13 H)

4.pooler层
    Pooler层的作用就是进一步处理这个[CLS]标记的向量，使其更适合作为分类任务的输入
    pooler层： H X H + H
    总参数量： H^2 + H

总参数量： (V + 516) X H + L X (12H^2 + 13 H) + H^2 + H

验证：V = 21128, H = 768, L = 1
总参数量： (V + 516) * H + L * (12 * H^2 + 13 * H) + H^2 + H
         (21128 + 516) X 768 + 1 X (12 * 768^2 + 13 * 768) + 768^2 + 768
'''
count = 0
for name, param in state_dict.items():
    # print(name, param.shape)
    z = param.shape
    try:
        # print(np.prod(z))
        count += np.prod(z)
    except:
        print('error')

V = 21128
H = 768
L = 1
print(f'\n 实际统计：Bert参数量transformer为{L}层时: ', count)
print(f'\n 理论计算：Bert参数量transformer为{L}层时: ', (V + 516) * H + L * (12 * H**2 + 13 * H) + H**2 + H)
#
# print(state_dict["embeddings.position_embeddings.weight"].numpy())
# print('\n\n')
# print(state_dict["embeddings.token_type_embeddings.weight"].numpy())