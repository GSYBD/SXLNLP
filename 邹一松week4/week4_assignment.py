# 定义字典；每个词后方存储的是其权重，此处权重为示例，可自行修改
Dict = {"自然": 0.2,
        "自然语言": 0.3,
        "语言": 0.1,
        "处理": 0.1,
        "技术": 0.2,
        "进步": 0.15,
        "迅速": 0.2,
        "处理技术": 0.25,
        "语": 0.05,
        "言处理": 0.15}

# 待切分文本
sentence = "自然语言处理技术进步迅速"

# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cuts(sentence, dict):
    target = []
    if not sentence:
        return [[]]  # 如果句子为空，返回空列表
    for i in range(1, len(sentence) + 1):
        word = sentence[:i]
        if word in dict:
            for sub_cut in all_cuts(sentence[i:], dict):
                target.append([word] + sub_cut)
    return target

# 获取分词结果中的最大权重组合
def get_max_weight_cut(cut_results, dict):
    max_weight = 0
    best_cut = None
    for cut in cut_results:
        weight = sum(dict.get(word, 0) for word in cut)
        if weight > max_weight:
            max_weight = weight
            best_cut = cut
    return best_cut, max_weight

# 执行全切分
cuts = all_cuts(sentence, Dict)

# 输出所有的切分方式
for cut in cuts:
    print(cut)

print("----------")

# 找出并输出权重最高的切分方式
best_cut, weight = get_max_weight_cut(cuts, Dict)
print("最高权重的切分方式：", best_cut, "权重值：", weight)
