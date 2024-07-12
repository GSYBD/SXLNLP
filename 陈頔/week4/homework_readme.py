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
def all_cut(sentence, Dict, prefix=None, results=None):
    if results is None:  
        results = []  # results存储所有可能的切分方式
    if prefix is None:  
        prefix = []  # prefix存储当前已经切分的词
    if not sentence:  # 如果文本为空，则代表当前句子已经切分完成  
        results.append(prefix.copy())  # 将当前切分添加到结果列表中  
        return  
    # 可理解为sentence[:i]与sentence[i:]把句子一分为二，
    # 在prefix中存储已经切分好的前半句词，在sentence[i:]中存储后半句待切分文本
    for i in range(1, len(sentence) + 1):  
        word = sentence[:i]  # sentence[:i]尝试长度为i的子串作为词  
        if word in Dict: 
            all_cut(sentence[i:], Dict, prefix + [word], results)  # 递归地处理剩余文本，sentence[i:]截取第i个字后面的所有
    return results

print('结果', all_cut(sentence, Dict), len(all_cut(sentence, Dict)))

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

