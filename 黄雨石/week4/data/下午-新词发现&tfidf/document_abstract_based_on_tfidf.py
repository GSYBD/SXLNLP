import jieba
import math
import os
import random
import re
import json
from collections import defaultdict
from calculate_tfidf import calculate_tfidf, tf_idf_topk
"""
基于tfidf实现简单文本摘要
"""

jieba.initialize()

#加载文档数据（可以想象成网页数据），计算每个网页的tfidf字典
def load_data(file_path):
    corpus = []
    with open(file_path, encoding="utf8") as f:
        documents = json.loads(f.read())
        for document in documents:
            assert "\n" not in document["title"]
            assert "\n" not in document["content"]
            corpus.append(document["title"] + "\n" + document["content"])
        tf_idf_dict = calculate_tfidf(corpus)
    return tf_idf_dict, corpus

#计算每一篇文章的摘要
#输入该文章的tf_idf词典，和文章内容
#top为人为定义的选取的句子数量
#过滤掉一些正文太短的文章，因为正文太短在做摘要意义不大
def generate_document_abstract(document_tf_idf, document, top=3):
    sentences = re.split("？|！|。", document)
    #过滤掉正文在五句以内的文章
    if len(sentences) <= 5:
        return None
    result = []
    for index, sentence in enumerate(sentences):
        sentence_score = 0
        words = jieba.lcut(sentence)
        for word in words:
            sentence_score += document_tf_idf.get(word, 0)
        sentence_score /= (len(words) + 1)
        result.append([sentence_score, index])
    result = sorted(result, key=lambda x:x[0], reverse=True)
    #权重最高的可能依次是第10，第6，第3句，将他们调整为出现顺序比较合理，即3,6,10
    important_sentence_indexs = sorted([x[1] for x in result[:top]])
    return "。".join([sentences[index] for index in important_sentence_indexs])

#生成所有文章的摘要
def generate_abstract(tf_idf_dict, corpus):
    res = []
    for index, document_tf_idf in tf_idf_dict.items():
        title, content = corpus[index].split("\n")
        abstract = generate_document_abstract(document_tf_idf, content)
        if abstract is None:
            continue
        corpus[index] += "\n" + abstract
        res.append({"标题":title, "正文":content, "摘要":abstract})
    return res


if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, corpus = load_data(path)
    res = generate_abstract(tf_idf_dict, corpus)
    writer = open("abstract.json", "w", encoding="utf8")
    writer.write(json.dumps(res, ensure_ascii=False, indent=2))
    writer.close()
