import jieba
import itertools

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

def all_cut(sentence, Dict):
    # 切分句子
    def backtrack(start, path):
        if start == len(sentence):
            targets.append(path)
            return
        for end in range(start+1, len(sentence)+1):
            word = sentence[start:end]
            if word in Dict or len(word) == 1:
                backtrack(end, path + [word])
    targets = []
    backtrack(0, [])
    return targets


print(all_cut(sentence, Dict))

target = [['经常', '有意见', '分歧'],
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