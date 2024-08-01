import jieba

text = "我爱北京天安门"
seg_list = jieba.cut(text, cut_all=True)
print(" / ".join(seg_list))


def simple_full_cut(text, dict_words):
    result = []
    n = len(text)
    for i in range(n):
        for j in range(i + 1, n + 1):
            word = text[i:j]
            if word in dict_words:
                result.append(word)
    return result


# 假设有一个非常简单的词典
dict_words = {"我", "爱", "北京", "天安门", "爱北京", "天安门广场"}

text = "我爱北京天安门"
print(" / ".join(simple_full_cut(text, dict_words)))
