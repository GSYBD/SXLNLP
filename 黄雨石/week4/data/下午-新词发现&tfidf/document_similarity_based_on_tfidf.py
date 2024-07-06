#coding:utf8
import jieba
import math
import os
import json
from collections import defaultdict
from calculate_tfidf import calculate_tfidf, tf_idf_topk

"""
基于tfidf实现文本相似度计算
"""

jieba.initialize()

#加载文档数据（可以想象成网页数据），计算每个网页的tfidf字典
#之后统计每篇文档重要在前10的词，统计出重要词词表
#重要词词表用于后续文本向量化
def load_data(file_path):
    corpus = []
    with open(file_path, encoding="utf8") as f:
        documents = json.loads(f.read())
        for document in documents:
            corpus.append(document["title"] + "\n" + document["content"])
    tf_idf_dict = calculate_tfidf(corpus)
    topk_words = tf_idf_topk(tf_idf_dict, top=5, print_word=False)
    vocab = set()
    for words in topk_words.values():
        for word, score in words:
            vocab.add(word)
    print("词表大小：", len(vocab))
    return tf_idf_dict, list(vocab), corpus


#passage是文本字符串
#vocab是词列表
#向量化的方式：计算每个重要词在文档中的出现频率
def doc_to_vec(passage, vocab):
    vector = [0] * len(vocab)
    passage_words = jieba.lcut(passage)
    for index, word in enumerate(vocab):
        vector[index] = passage_words.count(word) / len(passage_words)
    return vector

#先计算所有文档的向量
def calculate_corpus_vectors(corpus, vocab):
    corpus_vectors = [doc_to_vec(c, vocab) for c in corpus]
    return corpus_vectors

#计算向量余弦相似度
def cosine_similarity(vector1, vector2):
    x_dot_y = sum([x*y for x, y in zip(vector1, vector2)])
    sqrt_x = math.sqrt(sum([x ** 2 for x in vector1]))
    sqrt_y = math.sqrt(sum([x ** 2 for x in vector2]))
    if sqrt_y == 0 or sqrt_y == 0:
        return 0
    return x_dot_y / (sqrt_x * sqrt_y + 1e-7)


#输入一篇文本，寻找最相似文本
def search_most_similar_document(passage, corpus_vectors, vocab):
    input_vec = doc_to_vec(passage, vocab)
    result = []
    for index, vector in enumerate(corpus_vectors):
        score = cosine_similarity(input_vec, vector)
        result.append([index, score])
    result = sorted(result, reverse=True, key=lambda x:x[1])
    return result[:4]


if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, vocab, corpus = load_data(path)
    corpus_vectors = calculate_corpus_vectors(corpus, vocab)
    passage = "魔兽争霸"
    for corpus_index, score in search_most_similar_document(passage, corpus_vectors, vocab):
        print("相似文章:\n", corpus[corpus_index].strip())
        print("得分：", score)
        print("--------------")