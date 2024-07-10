import torch

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
# sentence: 待切分的字符串
# Dict: 字典，键为可以切分的字符串片段
# current: 当前切分结果的一部分，用于递归过程中传递
# target: 存储所有可能的切分方式的列表
def all_cut(sentence, Dict, current, target):
    # TODO
    # 如果sentence为空，表示找到了一个完整的切分方式
    if not sentence:
        target.append(current[:])
        return
    # 遍历sentence，尝试切分
    for i in range(1, len(sentence)+1):
        # 检查当前片段是否在字典的键中
        if sentence[:i] in Dict:
            # 如果在，递归调用自身，继续切分剩余部分
            current.append(sentence[:i])
            all_cut(sentence[i:], Dict, current, target)
            # 回溯，移除当前片段，尝试其他可能的切分
            current.pop()

def print_dict(sentence, Dict):
    target = []
    text_list = []
    all_cut(sentence, Dict, [], target)
    for item in target:
        text = " ".join(item)    # 将切分结果拼接成字符串
        text = text.split()      # 将字符串按空格切分成列表
        text_list.append(text)   # 将列表存入列表
    print(text_list)
print_dict(sentence, Dict)

#目标输出;顺序不重要
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

