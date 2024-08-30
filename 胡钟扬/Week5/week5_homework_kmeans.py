import json

import gensim

from gensim.models import Word2Vec
import math

import os

import jieba
import numpy as np

import sklearn
from sklearn.cluster import KMeans
from collections import defaultdict

from typing import List,Dict

# 第5周作业：实现基于kmeans的类内距离计算，筛选优质类别。



jieba.initialize

def load_word2vec_model(path):
    '''
     读取词向量模型
    '''
    model = Word2Vec.load(path)

    return model
    
    
    
    
def load_sentence(path):
    sentences = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.lcut(sentence)))

        print("获取句子数量：", len(sentences))
    return sentences




from typing import List
def sentences_to_vectors(sentences:List[str], model)->np.ndarray:
    '''
            文本向量化
    '''
    vectors = []
    for sentence in sentences:
        
        words = sentence.split()
        
        vector_size = model.vector_size
        vector = np.zeros((vector_size,))
        for word in words:
            
            try:
                vector += model.wv[word]
            
            except KeyError:
                vector += np.zeros((vector_size,))
        
        vector /= len(words)
        
        vectors.append(vector)      
        
    return np.array(vectors)


def calc_in_cluster_distance(label_sentence_vector_dict, kmeans:KMeans)->Dict[int,float]:
    
    print("============ cluster centroids =======================")
    # print(kmeans.cluster_centers_)
    
    centers:List[List[float]] = kmeans.cluster_centers_ 
    
    cluster_labels = kmeans.labels_
    
    
    result = defaultdict(int)
    
    for label, vectors in label_sentence_vector_dict.items():
        sum = 0
        
        center_vector = centers[label]
        
        for vector in vectors:
            sum+=calc_distance(vector, center_vector)
            
        result[label] = sum
        
    
    return result
    
    


def calc_distance(vec1, vec2):
    '''
        计算余弦距离
    '''
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    
    product = np.dot(vec1, vec2)
    
    vec1_bar = np.sqrt(np.sum(np.square(vec1)))
    vec2_bar = np.sqrt(np.sum(np.square(vec2)))
    
    distance = product / (vec1_bar*vec2_bar)
    # print("余弦距离为 ：",distance)
    
    return distance



def top_k_clusters(result:Dict[int,float], top_k = 5):
    
    print(sorted(result.items(), key=lambda x:x[1], reverse=True)[:top_k])
    
    result = dict(sorted(result.items(), key=lambda x:x[1], reverse=True)[:top_k])
    
    
    print("type(result) = ",type(result))
    # result = result[:top_k]
    # print("after sorting :")
    
    # for k,v in result.items():
    #     print(f"label:{k}, in_cluster_sum:{v}")
        
    return result
    



def print_clusters(sentence_label_dict):
    for label, sentences in sentence_label_dict.items():
        print(f' ============= 第{label+1}聚类 ===========')
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ",""))
        print("------------------------------------")


def main():
    '''
        
    '''
    
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    print("==================== sentences ================")
    print(sentences)
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    print(" ============= 所有标题向量化 ====================")
    print(vectors)
    
    cluster_num = int(math.sqrt(len(sentences)))
    
    kmeans = KMeans(cluster_num)
    kmeans.fit(vectors)
    
    
    
    sentence_label_dict = defaultdict(list)
    
    # 把同一个类的句子整理到一起
    for label, sentence in zip(kmeans.labels_, sentences):
        sentence_label_dict[label].append(sentence)
    
    
    # print_clusters(sentence_label_dict)
    
    
    
    # 构建1abel-sentenceVector dict
    label_sentence_vector_dict = defaultdict(list)
    
    for label, vector in zip(kmeans.labels_, vectors):
        label_sentence_vector_dict[label].append(vector)

    
    result = calc_in_cluster_distance(label_sentence_vector_dict, kmeans)
    
    
    
    result1 = top_k_clusters(result)
    
    
    print("topk 个 clusters：",sorted(result1.keys()))
    
    print("================ 以下是topk个聚类--根据类内距离进行排序 ===============")
    
    for label, sum in result1.items():
        sentences = sentence_label_dict.get(label)
        
        print(f" ================= cluster label {label} =================")
        
        for sentence in sentences:
            print(sentence.replace(" ", ""))
        
        
        
    print(" =================================================== ")
    print("topk 个 clusters：",sorted(result1.keys()))
    






if __name__ == '__main__':
    main()