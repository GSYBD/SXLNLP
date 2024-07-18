# 文本全切分
dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}


def all_cut(string, segment: list):
    result = []

    def cut(string, segment: list, result):
        if not string:
            result.append(segment)
        for i in range(1, len(string) + 1):
            # segment += [string[:i+1]]
            # string = string[i+1:]
            cut(string[i:], segment + [string[:i]], result)

    cut(string, segment, result)
    return result


string = "经常有意见分歧"
segment = []
target = []
results = all_cut(string,segment)
for result in results:
    flag = True
    for i in result:
        if i not in dict:
            flag = False
    if flag:
        target.append(result)

print(target)




