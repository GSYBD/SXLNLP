# week4作业

from collections import defaultdict

"""
实现给定文本，与切分词，实现文本的全切分，记录下来所有可能组成

广度优先的实现
word_dict：将切分词 组装成有向路径
cut_sentence： 顺着路径往前走，先找当前所有可能的路径，再往下找
"""

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


# 将词典组装成前缀树
# {'经': ['经常', '经'], '有': ['有', '有意见'], '常': ['常'], '歧': ['歧'], '意': ['意见', '意'], '分': ['分歧', '分'], '见': ['见', '见分歧']})
def word_dict(Dict):
    word_dict = defaultdict(list)
    for word, _ in Dict.items():
        if len(word) > 1:
            word_dict[word[0]].append(word)
        else:
            word_dict[word].append(word)
    return word_dict

# 递归找全部路径
def cut_sentence(sentence, word_map, path, target):
    if path:
        current_sentence = ''.join(path)
        if current_sentence[-1] == sentence[-1]:
            # 如果最后一个字相等，这一路径就找完，保存
            target.append(path)
            return target
        # 寻找下一个词
        next = sentence[len(current_sentence)]
    else:
        next = sentence[0]

    if next in word_map:
        for word in word_map[next]:
            # 如果字在词表里，就找完所有可能的路径，广度优先
            new_path = path[:]
            new_path.append(word)

            cut_sentence(sentence, word_map, new_path, target)

    return target


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    word_map = word_dict(Dict)

    targets = cut_sentence(sentence, word_map, [], [])

    return targets

target = all_cut(sentence, Dict)
for path in target:
    print(path)

# 目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]


