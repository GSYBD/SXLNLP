# -*- encoding: utf-8 -*-
'''
my_loader.py
Created on 2024/8/1 13:31
@author: Allan Lyu
@Description:
'''

import csv
import random

from torch.utils.data import Dataset, DataLoader


# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list of tuples): Each tuple contains (label, review).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


# 读取CSV文件
def read_csv(filepath):
    data = []
    with open(filepath, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row:  # 跳过空行
                label = row[0]  # 假设label是整数
                review = row[1]
                data.append((label, review))
    return data


# 数据预处理和切分
def split_dataset(data, train_ratio=0.8):
    random.shuffle(data)  # 打乱数据
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]
    return train_data, val_data


# 加载数据，切分数据集
def load_and_split_data(csv_filepath):
    # 读取CSV文件
    data = read_csv(csv_filepath)
    # 切分数据集
    train_data, val_data = split_dataset(data)
    return train_data, val_data


# 用torch自带的DataLoader类封装数据,并创建DataLoader
def load_data(csv_filepath, config):
    # 加载并切分数据
    train_data, val_data = load_and_split_data(csv_filepath)

    # 创建Dataset实例
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)

    # 创建DataLoader实例
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    # 使用示例
    my_csv_filepath = '../文本分类练习数据集/文本分类练习.csv'
    train_loader, val_loader = load_data(my_csv_filepath, config={"batch_size": 64})

    # # 验证数据加载效果
    # for batch in train_loader:
    #     labels, reviews = batch
    #     print(labels, reviews)
    #     break
    # for batch in val_loader:
    #     labels, reviews = batch
    #     print(labels, reviews)
    #     break
    # print(len(train_loader), len(val_loader))
    # print(len(train_loader.dataset), len(val_loader.dataset))

    # lyu: 打开一个文件,将train_loader的数据结果写到csv表, 要求: 每行数据格式为: label,review
    with open('../data/my_train_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for batch in train_loader:
            labels, reviews = batch
            for label, review in zip(labels, reviews):
                writer.writerow([label, review])

    # lyu: 打开一个文件,将val_loader的数据结果写到csv表, 要求: 每行数据格式为: label,review
    with open('../data/my_val_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for batch in val_loader:
            labels, reviews = batch
            for label, review in zip(labels, reviews):
                writer.writerow([label, review])
