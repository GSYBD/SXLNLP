

"""
week3作业：实现文本的全切分。
"""

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
# print(sentence[4:])
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict, path=[]):
    # 递归结束条件：当文本为空时，返回当前路径
    if not sentence:
        return [path]

    target = []
    # 遍历词典，尝试匹配文本开头的词
    for word in Dict:
        if sentence.startswith(word):
            # 如果找到匹配的词，递归地对剩余文本进行切分，并将当前词加入路径
            new_path = path + [word]
            # print(new_path)
            target.extend(all_cut(sentence[len(word):], Dict, new_path))

            # 如果没有找到匹配的词，则将单个字符作为词加入路径，并继续切分
    if not target:
        new_path = path + [sentence[0]]
        target.extend(all_cut(sentence[1:], Dict, new_path))

    return target

targets = all_cut(sentence, Dict)
# print(targets)
for seg in targets:
    print(seg)
    sum = 0.0
    for char in seg:
        sum += Dict[char]
    print("总词频: %.3f" %sum)

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
