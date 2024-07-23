# week3作业
import jieba

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
dictionary = {"经常": 0.1,
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
sentence = "经常有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式

def all_cut(sentence, dictionary):
    words = []
    length = len(sentence)
    for i in range(length):
        for j in range(i+1, length+1):
            word = sentence[i:j]
            if word in dictionary:
                words.append(word)
    return words

# 测试句子
sentence = "经常有意见分歧"
result = all_cut(sentence, dictionary)
result.sort()
print(result)



# 目标输出,顺序不重要
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

import numpy as np

arrays = [np.array(sublist) for sublist in target]

# 使用concatenate将它们合并
result = np.concatenate(arrays)

# 将NumPy数组转换为Python列表
res_list = result.tolist()
res = list(set(res_list))
res.sort()
print(res)
"""
['分', '分歧', '常', '意', '意见', '有', '有意见', '歧', '经', '经常', '见', '见分歧']
['分', '分歧', '常', '意', '意见', '有', '有意见', '歧', '经', '经常', '见', '见分歧']
"""
