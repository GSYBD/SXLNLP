# week3作业

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

# Input: 字典dict={w1, w2, ..., wn}
# Output: 有限状态机的pdict
def load_word_dict(dict):
    pdict = {}
    for word in dict:
        pdict[word] = 1
        lw = len(word)
        for x in reversed(range(1, lw)):  # x in {lw-1, lw-2, ..., 1}:
            subword = word[:x]
            if subword not in pdict:
                pdict[subword] = 0
            else:
                break
    return pdict


# Input: pdict, sentence
# Output: DAG
def DAG(pdict, sentence):
    # DAG = {}
    # N = len(sentence)
    # for i in range(0, N):  # {0, 1, ..., N-1}
    #     tmp_list = [i]
    #     j = i + 1
    #     subword = sentence[i:j + 1]
    #     while j < N and subword in pdict:
    #         if pdict[subword]:
    #             tmp_list.append(j)
    #     DAG[i] = tmp_list
    DAG = {}
    N = len(sentence)
    for i in range(0, N):  # {0, 1, ..., N-1}
        tmp_list = [i]
        j = i + 1
        while j < N:
            subword = sentence[i:j + 1]
            if subword in pdict:
                tmp_list.append(j)
            j += 1
        DAG[i] = tmp_list
    return DAG

# pdict = load_word_dict(Dict)
# print(DAG(pdict,"经常有意见分歧"))

# Input: DAG, sentence
# Output: words in possiable
def words_in_possible(DAG, sentence):
    N = len(sentence)
    words = []
    # print('N',N)
    for i in range(0, N):  # {0, 1, 2, ..., N}:
        tmp_list = DAG[i]
        # print('tmp_list',tmp_list)
        if len(tmp_list) > 1:
            words.extend([sentence[i:j+1] for j in tmp_list[0:]])
            # print('words',words)
        elif len(tmp_list) == 1:
            words.append(sentence[i])
            # print('words', words)
    return words


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    pdict = load_word_dict(Dict)
    dag = DAG(pdict, sentence)
    target = words_in_possible(dag, sentence)
    return target

# 待切分文本
sentence = "经常有意见分歧"
# 目标输出;顺序不重要
print(all_cut(sentence, Dict))

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
