import pandas as pd
from config import Config

data = pd.read_csv("dataset/文本分类练习.csv")
print("样本总数：", len(data))  # 11987
print("正样本数：", len(data[data["label"] == 1]))  # 4000
print("负样本数：", len(data[data["label"] == 0]))  # 7987

max_sequence_len, min_sequence_len, sum_sequence_len = 0, 100, 0
for i in range(len(data)):
    sequence_len = len(data["review"].iloc[i].strip())
    if sequence_len > max_sequence_len:
        max_sequence_len = sequence_len
    if sequence_len < min_sequence_len:
        min_sequence_len = sequence_len
    sum_sequence_len += sequence_len
print("最大句子长度：", max_sequence_len)  # 463
print("最小句子长度：", min_sequence_len)  # 5
print("平均句子长度：", sum_sequence_len / len(data))  # 25

data_0 = data[data["label"] == 0]
data_1 = data[data["label"] == 1]
train_size = Config["train_size"]
train_data_0 = data_0.sample(frac=train_size)
test_data_0 = data_0.drop(train_data_0.index)
train_data_1 = data_1.sample(frac=train_size)
test_data_1 = data_1.drop(train_data_1.index)
train_data = pd.concat([train_data_0, train_data_1, train_data_1], axis=0)
valid_data = pd.concat([test_data_0, test_data_1], axis=0)
train_data.to_csv("dataset/train_data.csv",index=False)
valid_data.to_csv("dataset/valid_data.csv",index=False)
