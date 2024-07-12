#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def all_cut(sentence, dict):
    results = []  # 初始化存储所有切分结果的列表
    # 基础情况：句子为空时，添加一个空列表作为有效切分
    if sentence == '':
        results.append([])
    else:
        for i in range(len(sentence)):  # 遍历句子以尝试所有可能的前缀
            if sentence[: i + 1] in dict:  # 检查前缀是否在词典中
                for sub_result in all_cut(sentence[i + 1:], dict):  # 对剩余部分递归进行全切分
                    results.append([sentence[: i + 1]] + sub_result)  # 将当前前缀与剩余部分的所有切分组合，添加到结果列表中
    return results


dict = {"经常": 0.1,
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

sentence = '经常有意见分歧'

if __name__ == '__main__':
    target = all_cut(sentence, dict)
    for each in target:
        print(each)

