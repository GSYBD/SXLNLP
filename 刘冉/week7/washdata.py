# -*- coding: utf-8 -*-

import csv
import os
import random
from config import Config
import json

class WashData:
    def __init__(self, config):
        self.file_path = "data/文本分类练习.csv"
        self.train_data_path = config["train_data_path"]
        self.valid_data_path = config["valid_data_path"]
        self.predict_data_path = config["predict_data_path"]
        if not os.path.exists(self.train_data_path):
            self.readCSVFile()
    def readCSVFile(self):
        with open(self.file_path, mode='r', encoding='utf-8') as file:
            # 创建CSV阅读器
            reader = csv.reader(file)
            # 将读取的文件内容乱序重拍 然后取一部分为训练数据 一部分为测试数据 一部分为推测数据
            rows = list(reader)
            all_data = rows[1:]#去掉第一条数据，第一条数据是标题栏
            random.shuffle(all_data)
            #文件总共11988 条 去掉首条为条目 [valid:600条] [predict:100条]剩下的为train
            print(len(all_data))
            print(max(len(row[1]) for row in all_data))#463最长文本
            train_array = all_data[1:len(all_data)-600-100]
            valid_array = all_data[len(train_array)+1:len(train_array)+1+600]
            #100条predict数据 不需要答案保存100个问题即可
            predict_array = list(input[1] for input in all_data[len(all_data)-100:])
            #训练数据保存为json文件
            with open(self.train_data_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(train_array, ensure_ascii=False, indent=4))
            #测试数据保存为json文件
            with open(self.valid_data_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(valid_array,ensure_ascii=False, indent=4))
            #predict数据 保存为json文件
            with open(self.predict_data_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(predict_array,ensure_ascii=False, indent=4))



if __name__ == "__main__":
    Wash(config=Config)