import csv
import random
import pandas as pd


def classic(path):
    with open(path, 'r', encoding='utf-8') as f:
        csvreader = csv.reader(f)
        header = next(csvreader)
        csvreader.line_num

        csv_list = []
        for row in csvreader:
            csv_list.append(row)
        random.shuffle(csv_list)
        tag_length = int(len(csv_list) * 0.8)
        train = csv_list[:tag_length]
        valid = csv_list[tag_length:]

        with open('data_train.csv', 'w', encoding='utf-8') as csv_file:
            fieldnames = [header[0], header[1]]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in train:
                writer.writerow({header[0]: row[0], header[1]: row[1]})

        with open('date_valid.csv','w',encoding='utf-8') as csv_file:
            fieldnames = [header[0], header[1]]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in valid:
                writer.writerow({header[0]: row[0], header[1]: row[1]})


classic('文本分类练习.csv')

# ---encoding:utf-8---
# @Author  : STZZ AIOT TLZS
# @Email   ：stzzaiottlzs@gmail.com
# @Site    : AIOT
# @File    : 逐行追加到csv文件.py
# @Software: PyCharm


def append_to_csv():
    data_df = pd.DataFrame(columns=["序号","a", "b"])
    data_df.to_csv("test.csv", mode="a", index=False, encoding="utf-8")

    data_df.loc[0] = ["1","1", "11"]
    data_df.to_csv("test.csv", mode="a", index=False, header=False,encoding="utf-8")

    data_df.loc[0] = ["2","2", "22"]
    data_df.to_csv("test.csv", mode="a", index=False, header=False, encoding="utf-8")


def __test_append_to_csv__():
    append_to_csv()


if __name__ == '__main__':
    __test_append_to_csv__()