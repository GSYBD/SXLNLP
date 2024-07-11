# 全切分，全切分的简单实现

import re
import time

def load_word_dict(path):
    max_word_length = 0
    word_dict = {}  #用set也是可以的。用list会很慢
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            word_dict[word] = 0
            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length

def full_segment(text, dic):
    word_list = []
    for i in range(len(text)):                  # i 从 0 到text的最后一个字的下标遍历
        for j in range(i + 1, len(text) + 1):   # j 遍历[i + 1, len(text)]区间
            word = text[i:j]                    # 取出连续区间[i, j]对应的字符串
            if word in dic:                     # 如果在词典中，则认为是一个词
                word_list.append(word)
    return word_list
 
 
if __name__ == '__main__':
    # 示例使用
    # word_dict = {"毕竟", "几人", "真得鹿", "得鹿", "不知终日", "不知", "终日", "梦为鱼"}
    # sentence = "毕竟几人真得鹿，不知终日梦为鱼"
    # word_dict = {
    # "我们", "在", "北京", "天安门", "广场", "上", "看", "升旗", "仪式", "随着", "自然", "显著的进展", "显著的",
    # "自然语言", "处理", "是", "计算机", "科学", "领域", "的", "一个", "分支", "显著", "进展", "发展", "情感分析",
    # "人工智能", "和", "人工", "智能", "深度学习", "技术", "正在", "推动", "这一", "等", "取得",
    # "的发展", "机器翻译", "机器", "情感"
    # }
    sentence = "随着人工智能技术的发展，自然语言处理在机器翻译、情感分析等领域取得了显著的进展"
    word_dict1, _ = load_word_dict("dict.txt")
    print(full_segment(sentence, word_dict1))

'''
['随', '随着', '着', '人', '人工', '人工智能', '工', '智', '智能', '能', '技', '技术', '术', '的', '发', '发展', '展', 
'自', '自然', '自然语言', '然', '语', '语言', '言', '处', '处理', '理', '在', '机', '机器', '机器翻译', '器', '翻', '翻译', '译', 
'情', '情感', '感', '分', '分析', '析', '等', '领', '领域', '域', '取', '取得', '得', '了', '显', '显著', '著', '的', '进', '进展', '展']

'''
