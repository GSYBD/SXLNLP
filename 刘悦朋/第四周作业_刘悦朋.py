import heapq
#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # 构造输出
    target = []

    # 拆分句子
    s = [char for char in sentence]

    # 若存在, 单字不存在于词表的情况, 则将其加入词表
    for char in s:
        if char not in Dict:
            Dict[char] = 0

    # 构造词典, 将每一个字按顺序赋值, 按待切分文本语序调整词表顺序
    dict = {}
    for char in s:
        dict[char] = len(dict)
    words = sorted(Dict.keys(), key=lambda w: dict[w[0]])

    # 记录词表中每个词的结束位置
    pos = []
    idx = 0
    for char in s:
        while words[idx][0] == char:
            heapq.heappush(pos, (dict[words[idx][-1]], words[idx]))
            idx += 1
            if idx == len(words):
                break

    # 因为每个单字都在词表中, 所以第一个字一定存在于词表中, 将其添加到输出
    _, char = heapq.heappop(pos)
    target.append([char])

    record = 1
    while pos:
        end, word = heapq.heappop(pos)
        if end == record:
            if len(word) == record + 1:
                target.append([word])
            for token in target:
                if len(''.join(token + [word])) == record + 1:
                    target.append(token + [word])
        else:
            heapq.heappush(pos, (end, word))
            record += 1
            if record == len(s):
                break

    target = [t for t in target if len(''.join(t)) == len(s)]

    return target

target = all_cut(sentence, Dict)
print(target)

# 输出结果
result = [['经', '常', '有', '意见', '分歧'],
          ['经常', '有', '意见', '分歧'],
          ['经', '常', '有意见', '分歧'],
          ['经常', '有意见', '分歧'],
          ['经', '常', '有', '意', '见', '分歧'],
          ['经常', '有', '意', '见', '分歧'],
          ['经', '常', '有', '意见', '分', '歧'],
          ['经常', '有', '意见', '分', '歧'],
          ['经', '常', '有意见', '分', '歧'],
          ['经常', '有意见', '分', '歧'],
          ['经', '常', '有', '意', '见', '分', '歧'],
          ['经常', '有', '意', '见', '分', '歧'],
          ['经', '常', '有', '意', '见分歧'],
          ['经常', '有', '意', '见分歧']
]

#目标输出;顺序不重要
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
