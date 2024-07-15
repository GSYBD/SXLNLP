def word_break(s, word_dict):
    if not s:
        return [[]]
    all_cuts = []
    for i in range(1, len(s) + 1):
        if s[:i] in word_dict:
            for rest_cuts in word_break(s[i:], word_dict):
                all_cuts.append([s[:i]] + rest_cuts)
    return all_cuts

sentence = "经常有意见分歧"
word_dict = {"经常","经","有","常","有意见","歧","意见","分歧","见","意","见分歧","分"}

target = word_break(sentence, word_dict)
print(target)
