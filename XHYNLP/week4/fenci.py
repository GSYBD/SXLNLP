def segment(text, word_dict):
    result = []

    def backtrack(start, path):
        if start == len(text):
            result.append(path[:])  
            return

        found_match = False
        for end in range(start + 1, len(text) + 1):
            word = text[start:end]
            if word in word_dict:
                found_match = True
                backtrack(end, path + [word])

        if not found_match and start < len(text):
            # 如果没有匹配词，将当前字符作为一个单词
            backtrack(start + 1, path + [text[start]])

    backtrack(0, [])
    return result

sentence = "北京大学生前来报到"
word_dict = {'北京', '大学', '北京大学', '大学生', '生前', '前来', '报到'}
results = segment(sentence, word_dict)
for r in results:
    print(" / ".join(r))

