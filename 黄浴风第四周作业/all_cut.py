#week3作业

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

#回溯函数
def backTrack(sentence, startIndex, path, result, Dict):
    if startIndex == len(sentence):
        result.append(path[:])
        return
    for endIndex in range(startIndex + 1, len(sentence) + 1):
        word = sentence[startIndex:endIndex]
        if word in Dict:
            path.append(word)
            backTrack(sentence, endIndex, path, result, Dict)
            path.pop()
    return 

def all_cut(sentence, Dict):
    result = []
    path = []
    backTrack(sentence, 0, path, result, Dict)
    return result
# 获取所有可能的切分方式
all_segmentations = all_cut(sentence, Dict)

# 输出结果
for seg in all_segmentations:
    print(seg)

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

