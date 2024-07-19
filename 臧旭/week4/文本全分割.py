def all_segmentations(sentence, dictionary):
    def segment(sentence, dictionary, memo):
        if sentence in memo:
            return memo[sentence]

        result = []
        if sentence in dictionary:
            result.append([sentence])

        for i in range(1, len(sentence)):
            prefix = sentence[:i]
            if prefix in dictionary:
                suffix = sentence[i:]
                suffix_segmentations = segment(suffix, dictionary, memo)
                for segmentation in suffix_segmentations:
                    result.append([prefix] + segmentation)

        memo[sentence] = result
        return result
    memo = {}
    return segment(sentence, dictionary, memo)


# 示例
Dict = {"经常", "经", "有", "常", "有意见", "歧", "意见", "分歧", "见", "意", "见分歧", "分"}
sentence = "经常有意见分歧"

segmentations = all_segmentations(sentence, Dict)
result = sorted(segmentations, key=lambda x: len(x))
print(result)
