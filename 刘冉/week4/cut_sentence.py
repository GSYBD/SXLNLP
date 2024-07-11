# week4作业

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
# 得到词典的词频最大长度
def get_key_max_length(Dict):
    max_len = 0
    for key in Dict:
        if len(key) > max_len:
            max_len = len(key)
    return max_len

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    max_len = get_key_max_length(Dict)
    target = cut_sentence(sentence, max_len, Dict)
    return target
# 递归遍历
def cut_sentence(sentence, max_len, Dict):
    if len(sentence) == 0:
        return []
    target = []
    #TODO
    for i in range(1, len(sentence) + 1):
        # 取出单词
        if i > max_len:
            break
        word = sentence[:i]
        # 如果单词在词典里 递归
        if word in Dict:
            cut_resluts = cut_sentence(sentence[i:], max_len, Dict)
            # 如果 cut_resluts 为空说明到最后了
            if len(cut_resluts):
                for cut_result in cut_resluts:
                    target.append([word] + cut_result)
            else:
                target.append([word])
    return target

# 得到词频最高的句子
def getHeightTF(sentence, Dict):
    target = all_cut(sentence, Dict)
    scores = []
    for result in target:
        score = 0
        for word in result:
            score += Dict[word]
        scores.append(score)
        print(result, 'score:', score)
    max_index = scores.index(max(scores))
    return target[max_index], max(scores)

target, score = getHeightTF(sentence, Dict)

print('最终结果：', target, 'score:', score)


'''
target = all_cut(sentence, Dict)
for result in target:
    print(result)

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

'''