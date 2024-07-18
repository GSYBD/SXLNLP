def full_segment(text, word_dict):
    n = len(text)
    dp = [[] for _ in range(n+1)]
    dp[0] = ['']  # 初始化，空字符串表示初始状态

    for i in range(1, n+1):
        for j in range(i):
            word = text[j:i]
            if word in word_dict and dp[j]:
                for segment in dp[j]:
                    dp[i].append((segment + ' ' + word).strip())

    return dp[n]

# 示例用法
text = "经常有意见分歧"
word_dict = {"经常", "经", "常", "有", "意见", "分歧", "有意见", "分", "歧"}

segments = full_segment(text, word_dict)
for segment in segments:
    print(segment)