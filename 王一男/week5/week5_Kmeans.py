#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 训练模型
def train_word2vec_model(corpus_path, dim=100):
    corpus = []
    with open(corpus_path, encoding="utf8") as f:
        for line in f:
            corpus.append(jieba.lcut(line))
    print(f"\n训练语料共有:{len(corpus)}条")
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    model.save("model.w2v")

# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            # print(jieba.cut(sentence))
            # print(" ".join(jieba.lcut(sentence)))
            sentences.add(" ".join(jieba.lcut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split() #sentence是分好词的，空格分开
        # return list
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))  # 加和求平均
    return np.array(vectors)  # index(corpus/line/sentence顺序) avg(word_vec)


def main():
    # train_word2vec_model(r"E:\111绝密资料，禁止外传(2)\AIML_llm\第五周 词向量\week5 词向量及文本向量\corpus.txt")
    model = load_word2vec_model(r"E:\111绝密资料，禁止外传(2)\AIML_llm\第五周 词向量\week5 词向量及文本向量\model.w2v")
    sentences = load_sentence(r"E:\111绝密资料，禁止外传(2)\AIML_llm\第五周 词向量\week5 词向量及文本向量\titles.txt")
    # for index, headline in enumerate(sentences):
    #     print(headline)
    #     if index == 11:
    #         break
    vectors = sentences_to_vectors(sentences, model)  # 语料行顺序, 值为词向量加和平均 100维向量*batches
    n_clusters = 2 * int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    vectors_label_dict = defaultdict(list)
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):
        sentence_label_dict[label].append(sentence)  # 存储label和语句
        vectors_label_dict[label].append(vector)   # 存储label和向量
    avg_label_distance = {}
    for label, vectors in vectors_label_dict.items():  # 计算平均距离导入label_distance字典
        cluster_center = kmeans.cluster_centers_[label]
        avg_label_distance[label] = np.mean([np.linalg.norm(vec - cluster_center) for vec in vectors])
    sorted_label_distance = sorted(avg_label_distance.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_label_distance[:10])  # 元组列表 (label, distance)
    k = 5
    top_k_label = [item[0] for item in sorted_label_distance[:k]]  # 取出前K个/根据平均距离判断保留数量?
    for label in top_k_label:
        print(f"cluster {label} :")
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))


if __name__ == "__main__":
    main()
