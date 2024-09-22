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

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    max_length = 0
    for key in Dict.keys() :
        key_length = len(key)
        if len(key) > max_length :
            max_length = key_length
    return rec_all_cut(sentence,Dict,max_length)

def rec_all_cut(sentence,Dict,max_length,idx = 0,sentence_cut = [],target = []):
    word = ''
    for _ in range(max_length) :
        word += sentence[idx]
        idx += 1
        if word in Dict:
            newSentence_cut = sentence_cut.copy()
            newSentence_cut.append(word)
            if idx >= len(sentence) :
                print(newSentence_cut)
                target.append(newSentence_cut)
            else:
                rec_all_cut(sentence,Dict,max_length,idx,newSentence_cut)
        if idx >= len(sentence) :
            break
    return target

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

all_cut(sentence,Dict)
