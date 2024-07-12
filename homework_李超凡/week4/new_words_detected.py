import math
import numpy as np


def get_corpus(corpus_path):
    # 读取语料数据
    text = ""
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            text += line.strip()
    return text


def get_word_counts(text, max_word_length):
    # n_gram统计词频,包括统计一个词左右两边相邻字符的频率
    word_count_dict = {}
    word_left_char_count = {}
    word_right_char_count = {}
    for word_length in range(1, max_word_length + 1):
        for i in range(len(text) - word_length - 1):
            word = text[i:i + word_length]
            if word in word_count_dict.keys():
                word_count_dict[word] += 1
            else:
                word_count_dict[word] = 1
                word_left_char_count[word] = {}
                word_right_char_count[word] = {}
            if i >= 1:
                left_char = text[i - 1]
                if left_char in word_left_char_count[word].keys():
                    word_left_char_count[word][left_char] += 1
                else:
                    word_left_char_count[word][left_char] = 0

            if i <= len(text) - word_length - 1:
                right_char = text[i + word_length + 1]
                if right_char in word_right_char_count[word].keys():
                    word_right_char_count[word][right_char] += 1
                else:
                    word_right_char_count[word][right_char] = 0
    return word_count_dict, word_left_char_count, word_right_char_count


def clc_word_score(word_count_dict, word_left_char_count, word_right_char_count):
    # 计算最终得分
    word_scores_dict = {}
    sum_count = sum(word_count_dict.values())
    for word in word_count_dict.keys():
        if len(word) >= 2:
            cohesion = clc_cohesion(word, word_count_dict, sum_count)
            left_entropy, right_entropy = clc_left_right_entropy(word, word_left_char_count, word_right_char_count)
            word_scores_dict[word] = cohesion * min(left_entropy, right_entropy)
    return word_scores_dict


def clc_cohesion(word, word_count_dict, sum_count):
    # 计算凝聚度
    p_x = word_count_dict[word] / sum_count
    p_y = 1
    for char in word:
        p_y *= word_count_dict[char] / sum_count
    return math.log(p_x / p_y, 2)


def clc_left_right_entropy(word, word_left_char_count, word_right_char_count):
    # 计算左右熵
    sum_count_left = sum(word_left_char_count[word].values())
    sum_count_right = sum(word_right_char_count[word].values())
    left_entropy = 0
    right_entropy = 0
    for key in word_left_char_count[word].keys():
        p_key = word_left_char_count[word][key] / sum_count_left
        left_entropy += -1 * p_key * math.log(p_key, 2)
    for key in word_right_char_count[word].keys():
        p_key = word_right_char_count[word][key] / sum_count_right
        right_entropy += -1 * p_key * math.log(p_key, 2)
    return left_entropy, right_entropy


def word_filter(word_score_dict, fliter_char):
    # 词过滤
    new_word_dict = {2: {"word": [], "score": []}, 3: {"word": [], "score": []}, 4: {"word": [], "score": []},
                     5: {"word": [], "score": []}}
    for word in word_score_dict.keys():
        if not any(char in word for char in fliter_char):
            new_word_dict[len(word)]["word"].append(word)
            new_word_dict[len(word)]["score"].append(word_score_dict[word])
    for key in new_word_dict:
        print(f"发现的{key}个字的新词有:",
              [new_word_dict[key]["word"][i] for i in list(np.array(new_word_dict[key]["score"]).argsort()[::-1][:10])])


if __name__ == "__main__":
    max_word_length = 5
    corpus_path = "sample_corpus.txt"
    fliter_char = ["，", "。", "-", "的", "、", "(", ")", " "]
    text = get_corpus(corpus_path)
    word_count_dict, word_left_char_count, word_right_char_count = get_word_counts(text, max_word_length)
    word_score_dict = clc_word_score(word_count_dict, word_left_char_count, word_right_char_count)
    word_filter(word_score_dict, fliter_char)
