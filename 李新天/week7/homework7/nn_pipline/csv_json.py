import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 读取 CSV 文件
csv_filename = '../data/文本分类练习.csv'
data = pd.read_csv(csv_filename, header=None, names=['label', 'review'])

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# 打乱数据
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 切分数据为训练集和验证集
train_data, valid_data = train_test_split(data, test_size=0.1, random_state=22)

train_list = train_data.to_dict(orient='records')
valid_list = valid_data.to_dict(orient='records')

with open('../data/train_tag_news.json', 'w', encoding='utf-8') as train_file:
    json.dump(train_list, train_file, ensure_ascii=False, indent=4)

with open('../data/valid_tag_news.json', 'w', encoding='utf-8') as valid_file:
    json.dump(valid_list, valid_file, ensure_ascii=False, indent=4)