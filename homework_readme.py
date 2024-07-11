#week3作业

# 词典；每个词后方存储的是其词频，词频在此处不使用
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

#递归实现，切分之后，递归切分剩余部分  回溯
def cut_sentence_recursive(sentence, path, results, start=0):
    if start == len(sentence):
        # 如果已经处理完整个句子，将当前结果添加到结果中
        results.append(path[:])
        return  # 不需要返回results，因为它在外部是可访问的

    for i in range(start + 1, len(sentence) + 1):
        # 尝试从start到i（不包括i）的子字符串作为单词
        word = sentence[start:i]
        if word in Dict:
            # 如果该单词在词典中，则继续递归处理剩余部分
            path.append(word)
            cut_sentence_recursive(sentence, path, results, i)
            path.pop()  # 回溯，移除最后一个添加的单词


if __name__ == "__main__":
    results = []
    sentence = "经常有意见分歧"
    cut_sentence_recursive(sentence, [], results)
    print(results)

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

