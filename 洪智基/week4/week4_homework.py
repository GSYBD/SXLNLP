#week4作业

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
def all_cut(sentence, Dict, start=0, current=[]):
        # 如果已经处理完整个句子
    if start == len(sentence):
        # 去除当前列表中可能的空字符串
        result = [word for word in current if word]
        if result:
            # 添加到结果列表中
            return [result]
        else:
            return []

    results = []
    # 遍历从当前位置开始的所有可能的词
    for i in range(start, len(sentence)):
        word = sentence[start:i + 1]
        # 如果这个词在词典中
        if word in Dict:
            # 递归处理剩余部分
            for next_results in all_cut(sentence, Dict, i + 1, current + [word]):
                results.append(next_results)

    return results

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

# 调用函数并打印结果
print(all_cut(sentence, Dict))