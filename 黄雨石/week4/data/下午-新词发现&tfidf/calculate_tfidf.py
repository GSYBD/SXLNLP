import jieba
import math
import os
import json
from collections import defaultdict

"""
tfidf的计算和使用
"""

#统计tf和idf值
def build_tf_idf_dict(corpus):
    tf_dict = defaultdict(dict)  #key:文档序号，value：dict，文档中每个词出现的频率
    idf_dict = defaultdict(set)  #key:词， value：set，文档序号，最终用于计算每个词在多少篇文档中出现过
    for text_index, text_words in enumerate(corpus):
        for word in text_words:
            if word not in tf_dict[text_index]:
                tf_dict[text_index][word] = 0
            tf_dict[text_index][word] += 1
            idf_dict[word].add(text_index)
    idf_dict = dict([(key, len(value)) for key, value in idf_dict.items()])
    return tf_dict, idf_dict

#根据tf值和idf值计算tfidf
def calculate_tf_idf(tf_dict, idf_dict):
    tf_idf_dict = defaultdict(dict)
    for text_index, word_tf_count_dict in tf_dict.items():
        for word, tf_count in word_tf_count_dict.items():
            tf = tf_count / sum(word_tf_count_dict.values())
            #tf-idf = tf * log(D/(idf + 1))
            tf_idf_dict[text_index][word] = tf * math.log(len(tf_dict)/(idf_dict[word]+1))
    return tf_idf_dict

#输入语料 list of string
#["xxxxxxxxx", "xxxxxxxxxxxxxxxx", "xxxxxxxx"]
def calculate_tfidf(corpus):
    #先进行分词
    corpus = [jieba.lcut(text) for text in corpus]
    tf_dict, idf_dict = build_tf_idf_dict(corpus)
    tf_idf_dict = calculate_tf_idf(tf_dict, idf_dict)
    return tf_idf_dict

#根据tfidf字典，显示每个领域topK的关键词
def tf_idf_topk(tfidf_dict, paths=[], top=10, print_word=True):
    topk_dict = {}
    for text_index, text_tfidf_dict in tfidf_dict.items():
        word_list = sorted(text_tfidf_dict.items(), key=lambda x:x[1], reverse=True)
        topk_dict[text_index] = word_list[:top]
        if print_word:
            print(text_index, paths[text_index])
            for i in range(top):
                print(word_list[i])
            print("----------")
    return topk_dict

def main():
    dir_path = r"category_corpus/"
    corpus = []
    paths = []
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        if path.endswith("txt"):
            corpus.append(open(path, encoding="utf8").read())
            paths.append(os.path.basename(path))
    tf_idf_dict = calculate_tfidf(corpus)
    tf_idf_topk(tf_idf_dict, paths)

if __name__ == "__main__":
    main()