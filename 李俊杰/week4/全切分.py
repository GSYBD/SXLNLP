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

# 待切分文本
sentence = "经常有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    n = len(sentence)
    results = []
    current_segmentation = []

    # 辅助函数，用于递归地进行全切分
    def backtrack(start, current_segmentation):
        # 如果当前开始位置等于句子的长度，添加当前切分结果
        if start == n:
            print(current_segmentation)
            results.append(current_segmentation)
            return 0

        for i in range(start + 1, n + 1):
            if sentence[start:i] in Dict:
                current_segmentation.append(sentence[start:i])
                backtrack(i, current_segmentation)
                current_segmentation.pop()

    # 从句子的开始位置开始递归全切分
    backtrack(0, current_segmentation)
    return results


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

# 调用函数
results = all_cut(sentence, Dict)
print(len(target), len(results))
# 打印结果
