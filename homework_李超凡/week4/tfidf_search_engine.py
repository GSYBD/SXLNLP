import jieba
import json
import math
import numpy as np


def load_corpus(corpus_path):
    # 加载json格式的数据
    corpus_ls = []
    with open(corpus_path, encoding="utf-8") as f:
        news = json.loads(f.read())
        for new in news:
            corpus_ls.append(new["title"] + "\n" + new["content"])
    return corpus_ls


def clc_tf_idf(corpus_ls):
    # 计算tf_idf
    tf_dic = {}
    idf_dic = {}
    words_ls = [jieba.lcut(corpus) for corpus in corpus_ls]
    for idx, words in enumerate(words_ls):
        tf_dic[idx] = {}
        for word in words:
            if word not in tf_dic[idx].keys():
                tf_dic[idx][word] = 0
            tf_dic[idx][word] += 1
        word_set = set(words)
        for word in word_set:
            if word not in idf_dic.keys():
                idf_dic[word] = 0
            idf_dic[word] += 1
    tf_idf_dic = tf_dic.copy()
    for idx in tf_dic.keys():
        for word in tf_dic[idx].keys():
            tf_idf_dic[idx][word] = tf_dic[idx][word] * math.log(len(corpus_ls) / (idf_dic[word] + 1), 2)
    return tf_idf_dic


def search_engine(query, corpus_ls, tf_idf_dic,top_number):
    # 搜索引擎
    word_ls = jieba.lcut(query)
    idx_ls = []
    score_ls = []
    for idx in tf_idf_dic.keys():
        score = 0
        for word in word_ls:
            if word in tf_idf_dic[idx].keys():
                score += tf_idf_dic[idx][word]
        if score != 0:
            idx_ls.append(idx)
            score_ls.append(score)
    sorted_index=list(np.argsort(np.array(score_ls)))[::-1]
    for i in range(top_number):
        print(corpus_ls[idx_ls[sorted_index[i]]])
        print("*"*20)


if __name__ == "__main__":
    corpus_path = "news.json"
    top_number = 3
    query="鸡茸鱼翅该怎么做？"
    corpus_ls = load_corpus(corpus_path)
    tf_idf_dic = clc_tf_idf(corpus_ls)
    search_engine(query, corpus_ls, tf_idf_dic, top_number)

