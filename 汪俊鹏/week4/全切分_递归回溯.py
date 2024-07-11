def word_break(s, wordDict):
    def backtrack(start, path):
        if start == len(s):
            result.append(' '.join(path))
            return

        for end in range(start + 1, len(s) + 1):
            if s[start:end] in wordDict:
                backtrack(end, path + [s[start:end]])

    result = []
    backtrack(0, [])
    return result

# 测试
s = "经常有意见分歧"
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
result = word_break(s, Dict)
for r in result:
    print(r)
