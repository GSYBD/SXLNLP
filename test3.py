import re
import time

#加载词典
Dict = {"经常": 0.1,
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

#待切分文本
sentence = "经常有意见分歧"

# def all_cut():
max_length = 0
for line in Dict:
        max_length = max(max_length, len(line))
lens = min(max_length, len(sentence))
def all_cut(sentence, Dict):
        inf = []
        def correct(start, cor_sen):
                if start == len(sentence):
                        if cor_sen != 0:
                                inf.append(cor_sen)
                        return

                for n in range(start, len(sentence) + 1):
                        word = sentence[start:n]
                        if word in Dict:
                                correct(n, cor_sen + [word])

        correct(0, [])

        return inf

inf = all_cut(sentence, Dict)
print(inf)

