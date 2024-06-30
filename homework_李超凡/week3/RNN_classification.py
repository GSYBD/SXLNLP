import torch
import torch.nn as nn
import numpy as np


# 构造数据集
def construct_dataset(dataset_path):
    # 构造数据集
    words_ls = []
    classification_ls = []
    label_ls = []
    max_words_len = 0
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            line_split = line.split()
            if len(line_split)>2:
                label_ls.append(line_split[0])
                classification_ls.append(line_split[1])
                words = "".join(line_split[2:])
                words_ls.append(words)
                if len(words) > max_words_len:
                    max_words_len = len(words)
    label_classification_dict = get_label_classification_dict(label_ls, classification_ls)
    vocabulary_dict = get_vocabulary(words_ls)
    x_arr = np.zeros((len(words_ls), max_words_len))
    for i in range(len(words_ls)):
        x_arr[i, -1 * len(words_ls[i]):] = np.array([vocabulary_dict[character] for character in words_ls[i]])
    print(x_arr)

    return words_ls, classification_ls, label_ls


def get_label_classification_dict(label_ls, classification_ls):
    # 获取label和类型标签
    label_classification_dict = {}
    for i in range(len(label_ls)):
        if label_ls[i] not in label_classification_dict.keys():
            label_classification_dict[label_ls[i]] = classification_ls[i]
    return label_classification_dict


def get_vocabulary(words_ls):
    # 获取词表
    vocabulary_dict = {"pad": 0}
    characters_ls = sorted(list(set("".join(words_ls))))
    for index, character in enumerate(characters_ls):
        vocabulary_dict[character] = index + 1
    vocabulary_dict["unk"] = len(vocabulary_dict)
    return vocabulary_dict


#     voc_dict=get_voc_dict(dataset_path)
#
#     pass
#
# def get_voc_dict(word_ls):

if __name__ == "__main__":
    dataset_path = "dataset/cnews_title/Train.txt"
    words_ls, classification_ls, label_ls = construct_dataset(dataset_path)
    # print(words_ls[:10])
