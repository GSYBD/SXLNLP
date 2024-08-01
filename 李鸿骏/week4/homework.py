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
target = []


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict, start=0, temp_sentence=[]):
    # 如果已经到达文本末尾，添加当前分词列表到结果列表
    if start == len(sentence):
        target.append(temp_sentence[:])  # 使用[:]来复制列表
        return
    # 尝试所有可能的词
    for i in range(start + 1, len(sentence) + 1):
        # 检查从start到i-1的子串是否在词典中
        if sentence[start:i] in Dict.keys():
            # 将当前词添加到路径中
            temp_sentence.append(sentence[start:i])
            # 递归调用，继续切分剩余的文本
            all_cut(sentence, Dict, i, temp_sentence)
            # 回溯，移除当前词
            temp_sentence.pop()
    return


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


def main():
    all_cut(sentence, Dict)
    print(target)
    print(len(target))
    return


if __name__ == "__main__":
    main()
