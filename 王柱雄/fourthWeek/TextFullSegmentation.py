def Text_full_segmentation(text, dictionary):
    """
    对输入文本进行全切分，并结合词语概率返回所有可能的分词结果。
    """
    if not text:
        return [[]]

    results = []

    for i in range(1, len(text) + 1):
        word = text[:i]
        if word in dictionary:
            # 对剩余部分进行全切分
            for sub_text in Text_full_segmentation(text[i:], dictionary):
                results.append(([word] + sub_text))

    return results


# 词典，包含词语及其概率
dictionary = {
    "每": 0.04,
    "一": 0.01,
    "天": 0.05,
    "一天": 0.05,
    "每一天": 0.05,
    "都": 0.03,
    "是": 0.02,
    "都是": 0.05,
    "个": 0.1,
    "一个": 0.3,
    "新": 0.05,
    "的": 0.1,
    "开始": 0.15
}

# 待切分的文本
text = "每一天都是一个新的开始"

# 进行全切分
result = Text_full_segmentation(text, dictionary)

# 所有可能的分词结果
for text in result:
    print(text)
