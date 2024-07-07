""""
实现分词的全排列
"""

Dict = {
    "经常": 0.1,
    "经": 0.1,
    "有": 0.1,
    "常": 0.1,
    "有意见": 0.1,
    "歧": 0.1,
    "意见": 0.1,
    "分歧": 0.1,
    "见": 0.1,
    "意": 0.1,
    "见分歧": 0.1,
    "分": 0.1,
}

# 文本
sentence = "经常有意见分歧"


# 实现全切分函数(14)
# 输出['经常'，xxx] [['经'，xxx],...]
def all_cut(sentence, dict):
    if len(sentence) < 1:
        return [[]]
    else:
        result = []
    for word in dict:
        if sentence.startswith(word):
            remaining = all_cut(sentence[len(word):], dict)
            for perm in remaining:

                result.append([word] + perm)

    return result


target = []
res = all_cut(sentence, Dict)
print(res)
print(len(res))


# python实现一个数组的全排列

def fun(list):
    if len(list) < 1:
        return [list]
    else:
        result = []
    for i in range(len(list)):
        remaining = list[:i] + list[i + 1:]
        for perm in fun(remaining):
            result.append([list[i]] + perm)
    return result

lst = [1, 2, 3]
res = fun(lst)
print(res)
