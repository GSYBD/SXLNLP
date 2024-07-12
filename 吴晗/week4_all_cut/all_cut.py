#week3作业
import jieba

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


def cut(sentence, Dict, seg, target):
    if sentence == '':
        target.append(seg)
    for i in range(0, len(sentence)):
        word = sentence[: i+1]
        if word in Dict:
            cut(sentence[i+1 :], Dict, seg+[word], target)


def all_cut(sentence, Dict):
    target = []
    seg = []
    cut(sentence, Dict, seg, target)
    return target


target = all_cut(sentence, Dict)
scores = []
for res in target:
    score = 0
    for word in res:
        score += Dict[word]
    scores.append(score)
    print(res, "\nscore: ", score)
max_id = scores.index(max(scores))
print('词频得分最高：', target[max_id], "score: ", max(scores))

# target, score = getHeightTF(sentence, Dict)

# print('最终结果：', target, 'score:', score)

# #目标输出;顺序不重要
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

