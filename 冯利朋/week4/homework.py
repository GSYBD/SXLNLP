

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


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict, target, current):
    if current is None:
        current = []
    if target is None:
        target = []
    if sentence == "":
        target.append(current[:])
        return
    for i in range(1, len(sentence) + 1):
        word = sentence[:i]
        if word in Dict:
            current.append(word)
            all_cut(sentence[i:], Dict, target, current)
            current.pop()
    return target


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

if __name__ == '__main__':
    results = all_cut(sentence, Dict, None, None)
    for re in results:
        print(re)
    print("----------根据词频选择最大的")
    # 根据词频，计算最大的输出
    res_dict = {index: sum([Dict[char] for char in cut_word]) for index, cut_word in enumerate(results)}
    # 排序
    sorted_val = sorted([(index, sum_val) for index, sum_val in res_dict.items()], key=lambda x: x[1], reverse=True)
    print(results[sorted_val[0][0]])
