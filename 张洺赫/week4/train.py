def full_cut(sentence, Dict):
    if not sentence:
        return [[]]
    
    all_cuts = []
    for i in range(1, len(sentence) + 1):
        word = sentence[:i]
        if word in Dict:
            for rest_cuts in full_cut(sentence[i:], Dict):
                rest_cuts.extend([word])
                all_cuts.append(rest_cuts)

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


if __name__ == "__main__":
    # 待切分文本  
    sentence = "经常有意见分歧"

    cuts = full_cut(sentence, Dict)
    for cut in cuts:
        print(" ".join(cut))