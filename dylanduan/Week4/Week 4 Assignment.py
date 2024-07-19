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

# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    def recursive_cut(sub_sentence):
        results = []
        for i in range(1, len(sub_sentence) + 1): # 循环递归
            prefix = sub_sentence[:i]       # 前缀
            if prefix in Dict:
                if i == len(sub_sentence):      # 到达末尾
                    results.append([prefix])
                else:
                    # 否则对剩余部分继续递归切分
                    for suffix_cut in recursive_cut(sub_sentence[i:]):
                        results.append([prefix] + suffix_cut)
        return results
    
    return recursive_cut(sentence)

result = all_cut(sentence, Dict)
for r in result:
    print(r)



#目标输出例子;顺序不重要
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

