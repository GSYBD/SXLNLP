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
#待切分文本
sentence = "经常有意见分歧"
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):

    def cut(sentence, Dict, step=1, parent = [], all = []):
        while step <= len(sentence):
            parent_copy = parent.copy()
            if sentence[0:step] in Dict:
                right_word = sentence[step:len(sentence)]
                parent_copy.append(sentence[0:step])
                if len(right_word) > 0:
                    cut(right_word, Dict, 1, parent_copy, all)
                else:
                    all.append(parent_copy)
            step += 1

    parent = []
    all = []
    cut(sentence, Dict, 1, parent, all)
    print(all)

all_cut(sentence, Dict)