# 曹哲鸣 第四周作业
#根据给定字典对文本进行全切分


import numpy as np

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

#获取字典中的最大长度
def GetMaxLength(dict):
    return max(len(key) for key in dict.keys())

#全切分
def CutAll(sentence, dict, max_length):
    target = []
    if len(sentence) == 0:
        return []
    for i in range(1, len(sentence) + 1):
        if i > max_length:
            break
        words = sentence[:i]
        if words in dict:
            cut_results = CutAll(sentence[i:], dict, max_length)
            if len(cut_results):
                for cut_result in cut_results:
                    target.append([words] + cut_result)
            else:
                target.append([words])
    return target

#获取频率最高的结果
def GetHeightTF(sentence, dict, max_length):
    target = CutAll(sentence, dict, max_length)
    score = []

    for line in target:
        scores = 0
        for word in line:
            scores += Dict[word]
        score.append(scores)
        print(line, scores)

    target_index = np.argmax(score)
    print("最终结果：", target[target_index], score[target_index])

max_length = GetMaxLength(Dict)

GetHeightTF(sentence, Dict, max_length)
