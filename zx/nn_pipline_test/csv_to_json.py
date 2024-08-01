import csv
import os
import random
from config import Config
import json

class toJson:
    def __init__(self, config):
        self.file_path = "../data_test/teat.csv"
        self.train_data_path = config["train_data_path"]
        self.valid_data_path = config["valid_data_path"]
        if not os.path.exists(self.train_data_path):
            self.readCSVFile()
    def readCSVFile(self):
        with open(self.file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
            all_data = rows[1:]
            random.shuffle(all_data)
            #文件总共11988 条 去掉首条为条目 [valid:900条] [predict:100条]剩下的为train
            print(len(all_data))
            print(max(len(row[1]) for row in all_data))#463最长文本
            train_array = all_data[1:len(all_data)-900]
            valid_array = all_data[len(train_array)+1:len(train_array)+1+900]
            #训练数据保存为json文件
            with open(self.train_data_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(train_array, ensure_ascii=False, indent=4))
            #测试数据保存为json文件
            with open(self.valid_data_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(valid_array,ensure_ascii=False, indent=4))



if __name__ == "__main__":
    toJson(config=Config)