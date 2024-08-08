# create_vocab.py
import pandas as pd
from collections import Counter
import json

# 加载数据
file_path = 'D:/BaiduNetdiskDownload/week7 文本分类问题/文本分类练习数据集/文本分类练习.csv'
data = pd.read_csv(file_path, header=None, names=['label', 'text'])

# 创建词汇表
def create_vocab(data, vocab_size=5000):
    all_words = []
    for text in data['text']:
        words = list(text.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace(',', ' ').replace('!', ' ').split())
        all_words.extend(words)
    word_counter = Counter(all_words)
    most_common_words = word_counter.most_common(vocab_size - 1)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common_words)}
    vocab["<PAD>"] = 0  # 添加一个填充符号
    return vocab

# 生成词汇表
vocab = create_vocab(data)
vocab_path = 'D:/BaiduNetdiskDownload/week7 文本分类问题/文本分类练习数据集/vocab.json'

# 保存词汇表
with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

print(f"词汇表已生成并保存到 {vocab_path}")
