# coding: utf-8

def all_cut(sentence, Dict):
    # 递归结束条件：如果句子为空，则添加一种切分方式（空列表）  
    if not sentence:
        return [[]]

    # 存储所有可能的切分方式
    all_cuts = []

    # 遍历所有可能的分割点  
    for i in range(1, len(sentence) + 1):
        # 取出当前分割点之前的词  
        word = sentence[:i]
        # 如果这个词在字典中  
        if word in Dict:
            # 递归调用，获取剩余部分的所有切分方式  
            for rest_cuts in all_cut(sentence[i:], Dict):
                # 将当前词与剩余部分的切分方式组合起来  
                current_cut = [word] + rest_cuts
                all_cuts.append(current_cut)

    return all_cuts


# 字典
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

# 待切分文本  
sentence = "经常有意见分歧"

cuts = all_cut(sentence, Dict)
for cut in cuts:
    print(" ".join(cut))