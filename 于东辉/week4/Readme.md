```
def all_cut(sentence, Dict):
    def dfs(start, path):
        if start == len(sentence):
            result.append(path[:])
            return
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                path.append(word)
                dfs(end, path)
                path.pop()  # 回溯，移除最后一个加入的词

    result = []
    dfs(0, [])
    return result

# 词典
Dict = {
    "经常": 0.1,
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
    "分": 0.1
}

# 待切分文本
sentence = "经常有意见分歧"

# 调用全切分函数
target = all_cut(sentence, Dict)
for item in target:
    print(item)
```