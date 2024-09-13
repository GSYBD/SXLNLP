import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# 加载数据
data = pd.read_csv('data/文本分类练习.csv')
data = shuffle(data)

train, valid = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
train.to_json('data/train.json', orient='records', force_ascii=False, lines='orient')
valid.to_json('data/valid.json', orient='records', force_ascii=False, lines='orient')





