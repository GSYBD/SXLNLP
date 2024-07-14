# 文本全切分

'''
Week4 Kivi
给定词表，一段语料
给出语料所有可能的切分结果

切分结果类似于树，从根节点到叶节点的一条路径就是一个切分结果
'''

def cut(string, dict, segments:list, target:list):
    # string: 当前要处理的文本
    # dict: 词表
    # segments: 当前步骤中已经切分的片段
    # target: 记录所有成果切分的结果
    if string == '':
        target.append(segments)
    # else:
    for i in range(0, len(string)):
        if string[: i+1] in dict:
            next_segments = segments + [string[: i+1]] # 将当前的片段添加到已经切分的片段中，传递给下层迭代
            cut(string[i+1 :], dict, segments + [string[: i+1]], target) # 迭代切分剩余的部分
                
def cut_all(sentence, dict):
    target = []
    segments = []
    cut(sentence, dict, segments, target)
    return target

dict = {"经常":0.1,
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

sentence = '经常有意见分歧'
target = cut_all(sentence, dict)
for each in target:
    print(each)
