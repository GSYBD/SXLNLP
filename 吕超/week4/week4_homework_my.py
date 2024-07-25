# -*- encoding: utf-8 -*-
'''
week4_homework_my.py
Created on 2024/7/11 23:50
@author: Allan Lyu
'''


"""
返回字符串s根据字典dictionary能够切分出的所有切分方式。
param s: 待切分的字符串
param dictionary: 字典，包含所有可能的单词
return: 所有可能的切分方式的列表，每个切分方式是一个单词列表
"""
def all_cut(s, dictionary):
    # 递归终止条件：如果字符串为空，则返回一个包含空字符串的列表作为一种切分方式
    if not s:
        return [[]]

    # 存储所有可能的切分方式的列表
    segmentations = []

    # 遍历所有可能的分割点
    for i in range(1, len(s) + 1):
        # 截取当前子串
        sub_string = s[:i]
        # 检查当前子串是否在字典中
        if sub_string in dictionary:
            # 递归地处理剩余部分
            for rest_segmentation in all_cut(s[i:], dictionary):
                # 将当前子串添加到所有剩余部分的切分方式之前
                segmentations.append([sub_string] + rest_segmentation)
    return segmentations


Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

# 遍历Dict获取key
Dict = list(Dict.keys())
# print(Dict)
sentence = "经常有意见分歧"

if __name__ == "__main__":
    # 调用函数进行切分
    targets = all_cut(sentence, Dict)
    # 列表按行打印
    for target in targets:
        print(target)
