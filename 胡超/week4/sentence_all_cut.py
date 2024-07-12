# -*- coding: utf-8 -*-
"""
author: Chris Hu
date: 2024/7/11
desc:
sample
"""
import time
# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

# 待切分文本
# sentence = "经常有意见分歧"
sentence = "经常分歧"
# 目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(input_sentence, input_dict):
    # 如果句子为空，返回一个空的切分列表
    if not input_sentence:
        return [[]]

    # 存储所有可能的切分结果
    result = []

    # 对于句子中的每一个可能的切分位置
    for i in range(1, len(input_sentence) + 1):
        # 检查当前的前缀是否在词典中
        prefix = input_sentence[:i]
        if prefix in input_dict:
            # 递归地切分剩余的部分
            suffix_cuts = all_cut(input_sentence[i:], input_dict)
            # 将前缀和剩余部分的所有可能切分组合起来
            for cut in suffix_cuts:
                result.append([prefix] + cut)
    return result


def optimal_all_cuts(input_sentence, input_dict):
    # 因为不考虑权重，故转换成集合以便快速查找
    dictionary_set = set(input_dict)

    def dp(start):
        # 如果已经到达句子末尾，返回一个空切分列表
        if start == len(input_sentence):
            return [[]]

        # 如果这个位置已经计算过，直接返回结果
        if start in memo:
            return memo[start]

        # 初始化结果列表
        res = []

        # 尝试从当前位置开始的所有可能的切分
        for end in range(start + 1, len(input_sentence) + 1):
            word = input_sentence[start:end]
            if word in dictionary_set:
                # 对于剩下的部分递归调用dp，然后把当前单词加到前面
                for sub_res in dp(end):
                    res.append([word] + sub_res)

        # 记忆化存储结果
        memo[start] = res
        return res

    # 初始化记忆化字典
    memo = {}
    return dp(0)


if __name__ == '__main__':
    start1 = time.perf_counter()
    brute_force_result = all_cut(sentence, Dict)
    end1 = time.perf_counter()

    start2 = time.perf_counter()
    optimal_result = optimal_all_cuts(sentence, Dict)
    end2 = time.perf_counter()
    print(f"**暴力解法**\n总耗时：{(end1 - start1)*1000:.2f}ms，切分结果如下:\n{brute_force_result}\n")
    print(f"**优化解法**\n总耗时：{(end2 - start2)*1000:.2f}ms，切分结果如下:\n{optimal_result}")
