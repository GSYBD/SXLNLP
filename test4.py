import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import operator

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

model = load_word2vec_model(r"D:\BaiduNetdiskDownload\八斗精品班\第五周 词向量\week5 词向量及文本向量\model.w2v")  #加载词向量模型
'''
print(model)    Word2Vec<vocab=19322, vector_size=100, alpha=0.025> 词表数量 词向量维度 学习率
'''
sentences = load_sentence(r"D:\BaiduNetdiskDownload\八斗精品班\第五周 词向量\week5 词向量及文本向量\titles.txt")   #加载所有标题
vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
print("指定聚类数量：", n_clusters)
kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
kmeans.fit(vectors)         # 进行聚类运算
# print(sentences)
# print(kmeans.labels_)
center = kmeans.cluster_centers_    #计算聚类中心

sentence_label_dict = defaultdict(list)
center_distance = defaultdict(list)

for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):
    sentence_label_dict[label].append(sentence)
    distance = np.linalg.norm(vector - center[label])   # 计算多维向量的欧氏距离
    center_distance[label].append(distance)

# 计算平均距离
avg_distance = {}
for label, distances in center_distance.items():
    # print(np.mean(distances))       # 多维向量取平均
    avg_distance[label] = np.mean(distances)
print(avg_distance)
# 筛选优质类别
excellent_distance = sorted(avg_distance.items(), key=lambda x: x[1], reverse=False)

for key in excellent_distance:
    print("cluster %d --- avg_distance: %f:" % (key[0], avg_distance[key[0]]))