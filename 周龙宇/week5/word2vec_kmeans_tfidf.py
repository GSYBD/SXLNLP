#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from tfidf import calculate_tfidf
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def generate_document_abstract(document_tf_idf, sentences, top=3):
    # sentences = re.split("？|！|。", document)
    #过滤掉正文在五句以内的文章
    if len(sentences) <= 5:
        return None
    result = []
    for index, sentence in enumerate(sentences):
        sentence.replace(" ", "")
        sentence_score = 0
        words = jieba.lcut(sentence)
        for word in words:
            sentence_score += document_tf_idf.get(word, 0)
        sentence_score /= (len(words) + 1)
        result.append([sentence_score, index])
    result = sorted(result, key=lambda x:x[0], reverse=True)
    #权重最高的可能依次是第10，第6，第3句，将他们调整为出现顺序比较合理，即3,6,10
    important_sentence_indexs = sorted([x[1] for x in result[:top]])
    return " | ".join([sentences[index].replace(" ", "") for index in important_sentence_indexs])

def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    corpus = []
    corpus.append(open("titles.txt", encoding="utf8").read())
    # print(corpus)
    # tfidf = calculate_tfidf(corpus)
    # print(tfidf)
    sentences = load_sentence("titles.txt")  #加载所有标题
    # print(sentences)
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    print(vectors.shape)
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    k_center = kmeans.cluster_centers_
    print("聚类中心：", len(k_center))
    print("聚类标签", len(kmeans.labels_))

    sentence_label_dict = defaultdict(list)
    sentence_vector_dict = defaultdict(list)
    for sentence, label, index in zip(sentences, kmeans.labels_, range(len(kmeans.labels_))):  #取出句子和标签

        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        sentence_vector_dict[label].append(vectors[index])
    # print(len(sentence_vector_dict[0]))
    # print(sentence_label_dict)
    corpus = [''.join(i).replace(" ", "") for i in sentence_label_dict.values()]
    tfidf = calculate_tfidf(corpus)
    # print(tfidf)
    sentence_os_dis_dict = defaultdict(list)
    for label, sentences in sentence_label_dict.items():
        tmp_dis = []
        for i in range(len(sentences)):
            tmp_dis.append(np.linalg.norm(k_center[label] - vectors[i]))
        sentence_os_dis_dict[label].append(np.mean(tmp_dis))  #计算距离，存入字典

    rerank_ = sorted(sentence_os_dis_dict.items(), key=lambda x:x[1])
    # print(rerank_)
    for index, i in enumerate(rerank_):
        if index > 10:
            break
        print("\n cluster : %s " % i[0])
        print("欧式距离是 : %.3f" % i[1][0])
        # print(sentence_label_dict[i[0]])
        tnp = generate_document_abstract(tfidf, sentence_label_dict[i[0]])
        print(f"摘要内容是 : {tnp}")
        print("----------------------------------------------------------------------------")

if __name__ == "__main__":
    main()

