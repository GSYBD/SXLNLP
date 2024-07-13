

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
    if not sentence:
        return [[]]

    target = []
    for i in range(1, len(sentence) + 1):
        word = sentence[:i]
        if word in Dict:
            remaining_sentence = sentence[i:]
            remaining_cuts = all_cut(remaining_sentence, Dict)
            for cut in remaining_cuts:
                target.append([word] + cut)

    return target

result = all_cut(sentence, Dict)
print(result)