#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import requests
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""

基于tfidf实现简单文本摘要（500字的摘要）
然后根据tfidf 进行kmeans分类
筛选优质类别
 后续可以拿着kmeans的分类去做模型训练（Bert模型输入有限制）

"""
#-----------------文本摘要-----------------
jieba.initialize()
# 加载停顿次
url = "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt"
response = requests.get(url)
chinese_stop_words = response.text.splitlines()
# 加载文档数据，取出标题和文章内容
def load_data(file_path):
    contents = []
    with open(file_path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) < 10:
                continue
            contents.append(line)
    return contents

contents = load_data('train.txt')
#jieba 分词
tokenized_contents = [' '.join(jieba.cut(content)) for content in contents]
# 使用TF-IDF向量化器 去除停用词
vectorizer = TfidfVectorizer(stop_words=chinese_stop_words)
tfidf_matrix = vectorizer.fit_transform(tokenized_contents)
feature_names = vectorizer.get_feature_names_out()
# 对于每篇文章，提取摘要
def extract_summary(doc_index, max_length=500):
    # 根据文档索引获取TF-IDF矩阵中对应文档的TF-IDF向量，并转换为密集矩阵形式
    doc_tfidf = tfidf_matrix[doc_index].todense()
    sentence_scores = []
    # 分割文档为句子，为计算每个句子的TF-IDF值和相似度分数做准备
    sentences = re.split("？|！|。", contents[doc_index])
    for sentence in sentences:
        if sentence:
            # 计算单个句子的TF-IDF向量，并转换为密集矩阵形式
            sentence_tfidf = vectorizer.transform([sentence]).todense()
            # 计算文档TF-IDF向量与句子TF-IDF向量的余弦相似度
            score = cosine_similarity(np.asarray(doc_tfidf), np.asarray(sentence_tfidf))[0][0]
            sentence_scores.append((sentence, score))
    # 根据相似度分数对句子进行排序，以便选择排名较高的句子作为摘要
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    # 构建文章摘要，直到达到预设的最大长度
    summary = ''
    for sentence, _ in sorted_sentences:
        if len(summary) == 0:
            summary += sentence + '.'
        elif len(summary) + len(sentence) < max_length:
            summary += sentence + '.'
        else:
            break

    return summary
# 获取文章摘要，做kmeans分类
def get_summaries(contents=contents):
    summaries = []
    for i in range(len(contents)):
        summary = extract_summary(i)
        summaries.append(" ".join(jieba.cut(summary)))
    return summaries

#------------------kmeas分类，可以用TFIDF,也可以用W2V，这里用了W2V----------------------
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

#---------------作业2种方法-----------------------
#1.按照距离排序-与label_center比较更近的距离放前面
def sort_by_distance(kmeans, vectors, sentences):
    sentence_vectors_dict = {k: v for k, v in zip(sentences, vectors)}
    # defaultdict 字典值不存在默认为[]
    sentence_label_dict = defaultdict(list)
    distance_label_dict = defaultdict(list)#为了记录距离 看数据是否准确
    for sentence, label in zip(sentences, kmeans.labels_):
        #按距离插入 同标签的放到一起
        label_center = kmeans.cluster_centers_[label]
        if len(sentence_label_dict[label]) > 0:
            insert_index = -100
            distance = 1e-3
            for index, sentence_label in enumerate(sentence_label_dict[label]):
                vector = sentence_vectors_dict[sentence]
                vector_label = sentence_vectors_dict[sentence_label]
                #用np的np.linalg.norm计算距离
                distance = np.linalg.norm(vector - label_center)
                distance_label = np.linalg.norm(vector_label - label_center)
                if distance < distance_label:
                    insert_index = index
                    break
            if insert_index == -100:
                sentence_label_dict[label].append(sentence)  #数组内的距离都比要插入的小
                distance_label_dict[label].append(distance)
            else:
                sentence_label_dict[label].insert(insert_index, sentence)#在索引处 插入
                distance_label_dict[label].insert(insert_index, distance)
        else:
            vector = sentence_vectors_dict[sentence]
            distance = np.linalg.norm(vector - label_center)
            sentence_label_dict[label].append(sentence)#label没有数据直接插入
            distance_label_dict[label].append(distance)
    return sentence_label_dict, distance_label_dict

#2.先求出所有跟label_center的距离求平均，然后看离平均距离最近的
def sort_by_distance_avg(kmeans, vectors, sentences):
    sentence_vectors_dict = {k: v for k, v in zip(sentences, vectors)}
    sentence_label_dict = defaultdict(list)
    distance_label_dict = defaultdict(list)

    #先求出所有向量跟label_center的距离
    for sentence, label in zip(sentences, kmeans.labels_):
        label_center = kmeans.cluster_centers_[label]
        vector = sentence_vectors_dict[sentence]
        distance = np.linalg.norm(vector - label_center)
        sentence_label_dict[label].append(sentence)
        distance_label_dict[label].append(distance)

    # 按照distance的平均最小值跟最小的距离排序
    sorted_sentences = defaultdict(list)
    sorted_distances = defaultdict(list)
    for label in distance_label_dict:
        distances = distance_label_dict[label]
        sentences = sentence_label_dict[label]
        avg_distance = np.mean(distances)
        # 计算每个句子与label_center的距离差，然后跟label_center的距离差最小的句子
        distance_diff_sentence = [(abs(d - avg_distance), sentence) for d, sentence in zip(distances, sentences)]
        # 按最小距离排序
        distance_diff_sentence.sort(key=lambda x: x[0])
        
        sorted_sentences[label] = [item[1] for item in distance_diff_sentence]
        sorted_distances[label] = [item[0] for item in distance_diff_sentence]

    return sorted_sentences, sorted_distances

def main():
    model = load_word2vec_model(r"/Users/liuran/Desktop/SXLNLP/刘冉/week5/model.w2v") #加载词向量模型
    sentences = get_summaries() #加载所有内容
    vectors = sentences_to_vectors(sentences, model)   #将所有内容向量化

    # n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    n_clusters = 12
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict, distance_label_dict = sort_by_distance(kmeans, vectors, sentences)
    #打印文字 和 距离
    for (label, sentence), (label1, distances) in zip(sentence_label_dict.items(), distance_label_dict.items()):
        print("cluster %s :" % label)
        for i in range(min(10, len(sentence))):  #随便打印10个，太多了看不过来
            print(sentence[i].replace(" ", "") , distances[i])
        print("------------")

    sentence_dict, distance_dict = sort_by_distance_avg(kmeans, vectors, sentences)
     #打印文字 和 距离
    for (label, sentence), (label1, distances) in zip(sentence_dict.items(), distance_dict.items()):
        print("=======cluster %s :" % label)
        for i in range(min(10, len(sentence))):  #随便打印10个，太多了看不过来
            print(sentence[i].replace(" ", "") , distances[i])
        print("------------")

if __name__ == "__main__":
    main()

