def all_segmentations(sentence, word_dict):
    n = len(sentence)
    dp = [[] for _ in range(n + 1)]
    dp[0] = [[]]

    for i in range(n):
        for j in range(i + 1, n + 1):
            word = sentence[i:j]
            if word in word_dict:
               for segmentation in dp[i]:
                    dp[j].append(segmentation + [word])

    return dp[n]


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

all_segmentations = all_segmentations(sentence, Dict)
print(all_segmentations)

