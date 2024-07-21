Dict = {
    "经常": 0.1,
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
    "分": 0.1
}

# 待切分文本
sentence = "经常有意见分歧"


def all_cut(sentence, Dict):
    def helper(start):
        if start == len(sentence):
            return [[]]  # 完成切分，返回一个空列表
        result = []
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                for sublist in helper(end):
                    result.append([word] + sublist)
        return result

    return helper(0)


# 调用函数并打印结果
result = all_cut(sentence, Dict)
for r in result:
    print(r)

