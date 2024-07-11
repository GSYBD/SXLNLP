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

def full_segment(text,word_freq_dict):
    if not text:
        return [[]]
    return [[text[0]] + seg for seg in full_segment(text[1:],word_freq_dict)
            for word in word_freq_dict if text.startswith(word) and word_freq_dict[word]>0] or [[]]
def full_segment2(text,word_dict):
    if not text:
        return [[]]
    results=[]
    for i in range(1,len(text)+1):
        if text[:i] in word_dict:
            for reset_seg in full_segment2(text[i:],word_dict):
                results.append([text[:i]]+reset_seg)
    return results
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = full_segment(sentence,Dict)
    return target

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
def main():
    target=full_segment2(sentence,Dict)
    # print(type(target))
    for item in target:
        print(item)

if __name__ == "__main__":
    main()

