# week4作业

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
sentence = "经常有意见分歧"

res = []
path = []


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = []
    backtracking(sentence, [], target, Dict)
    return target


"""
sentence: 表示待划分的字符串
current: 表示当前划分的字符串
target: 表示已划分好的字符串集合
"""


def backtracking(sentence, current, target, Dict):
    if not sentence:
        target.append(current[:])
        return
    for i in range(1, len(sentence) + 1):
        s = sentence[:i]
        if s in Dict:
            current.append(s)
            backtracking(sentence[i:], current, target, Dict)
            current.pop()




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

print(all_cut(sentence, Dict))
print(len(all_cut(sentence, Dict)))
