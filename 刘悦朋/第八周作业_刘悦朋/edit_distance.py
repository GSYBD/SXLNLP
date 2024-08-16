import numpy as np
import math


# 编辑距离
def edit_distance(string1, string2):
    # len(string1) < len(string2)
    if len(string1) > len(string2):
        return edit_distance(string2, string1)
    min = len(string2)
    # 偏移 = 短文本减去长度差 / 2后向下取整, 最小为0
    bias = max(math.floor((len(string1) - (len(string2) - len(string1))) / 2), 0)
    # range (-bias, len(string2) + bias)
    target = ' ' * bias + string2 + ' ' * bias
    record = None
    for i in range(len(target) - len(string1)):
        init = ' ' * i + string1 + (len(target) - i - len(string1)) * ' '
        score = 0
        for j in range(len(init)):
            if init[j] == target[j]:
                continue
            else:
                score += 1
        if score < min:
            min = score
            record = init
    if record:
        print(min)
        print(record)
        print(target)
        print('------------------------')
    else:
        print(min)
        print(string1)
        print(string2)
        print('------------------------')
    return min


# 基于编辑距离的相似度
def similarity_based_on_edit_distance(string1, string2):
    return 1 - edit_distance(string1, string2) / max(len(string1), len(string2))


if __name__ == '__main__':
    string1 = 'aaaaa'
    string2 = 'bbbbbbb'

    edit_distance(string1, string2)

    string1 = 'sgaseat'
    string2 = 'g'

    edit_distance(string1, string2)

    string1 = 'asdfasf'
    string2 = 'dsgsegseg'

    edit_distance(string1, string2)

    string1 = 'sjdfs'
    string2 = 'ailsmdpse'

    edit_distance(string1, string2)
