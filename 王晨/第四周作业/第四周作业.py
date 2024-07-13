def all_cut(sentence, Dict):
    def cut(str, path, result):
        if not str:
            result.append(path)
            # print(result, path)
            return
        for i in range(1, len(str) + 1):
            word = str[:i]
            if word in Dict:
                cut(str[i:], path + [word], result)

    result = []
    cut(sentence, [], result)
    return result

# 示例测试
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

sentence = "经常有意见分歧"
result = all_cut(sentence, Dict)
print(result)