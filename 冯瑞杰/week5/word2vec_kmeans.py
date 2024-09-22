#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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
        words = sentence.split()  
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

def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   

    n_clusters = int(math.sqrt(len(sentences)))  
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  
    kmeans.fit(vectors)          

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_): 
        sentence_label_dict[label].append(sentence)        

    #计算类内距离
    density_dict = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vectors[vector_index]   
        center = kmeans.cluster_centers_[label]  
        distance = cosine_distance(vector, center)  
        density_dict[label].append(distance)    
    for label, distance_list in density_dict.items():
        density_dict[label] = np.mean(distance_list)   
    density_order = sorted(density_dict.items(), key=lambda x:x[1], reverse=True)  

    for label, distance_avg in density_order:
        print("cluster %s , avg distance %f: " % (label, distance_avg))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  
            print(sentences[i].replace(" ", ""))
        print("---------")

#向量余弦距离
def cosine_distance(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))  
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))  
    return np.sum(vec1 * vec2)

def eculid_distance(vec1, vec2):
    return np.sqrt((np.sum(np.square(vec1 - vec2))))

if __name__ == "__main__":
    main()