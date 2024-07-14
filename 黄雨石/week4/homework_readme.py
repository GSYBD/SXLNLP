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
    def _cut(sentence, Dict, current_cut, target):
        if not sentence:
            target.append(current_cut)
            return
        for i in range(1, len(sentence) + 1):
            word = sentence[:i]
            if word in Dict:
                _cut(sentence[i:], Dict, current_cut + [word], target)

    target = []
    _cut(sentence, Dict, [], target)

    return target


def all_cut2(sentence, Dict):
    def _cut(sentence, Dict, memo):
        if sentence in memo:
            return memo[sentence]

        if not sentence:
            return [[]]  # Return a list containing an empty list to signify end of sentence

        result = []
        max_length = max(len(word) for word in Dict)  # Find the length of the longest word in the dictionary

        for i in range(1, min(max_length + 1, len(sentence) + 1)):
            word = sentence[:i]
            if word in Dict:
                for sublist in _cut(sentence[i:], Dict, memo):
                    result.append([word] + sublist)

        memo[sentence] = result
        return result

    memo = {}
    return _cut(sentence, Dict, memo)


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

# print(all_cut(sentence,Dict))
print(all_cut2(sentence,Dict))