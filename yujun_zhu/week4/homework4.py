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
    def recurrent_function(string, word_dict, max_len, current_segmentation, all_segmentations):
        if not string:
            all_segmentations.append(current_segmentation[:])
            return
        
        for i in range(1, min(max_len, len(string)) + 1):
            word = string[:i]
            if word in word_dict:
                current_segmentation.append(word)
                recurrent_function(string[i:], word_dict, max_len, current_segmentation, all_segmentations)
                current_segmentation.pop()

    target = []
    word_dict = Dict.keys()
    max_len = max([len(word) for word in word_dict])
    
    recurrent_function(sentence, word_dict, max_len, [], target)
    
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

def main():
    #调用全切分函数
    result = all_cut(sentence, Dict)
    #检查结果
    print(result)
    if  set([tuple(x) for x in result]) == set([tuple(x) for x in target]):
        print("测试通过")

if __name__ == "__main__":
    main()