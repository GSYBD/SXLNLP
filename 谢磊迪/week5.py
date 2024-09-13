from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import jieba
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')


def create_word_vector(path):
    word_vector=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            word_vector.append(jieba.lcut(line.replace('\n','').replace(' ','')))
    return word_vector

def  word_vector_model(path,vector_size):
    word_vector = create_word_vector(path)
    model =Word2Vec(word_vector,vector_size=vector_size, min_count=2,sg=0)
    print(1)
    model.save('word2vec.ptl')


def  sentence_vector(path):
    model= KeyedVectors.load_word2vec_format('word2vec2.ptl', binary=True)
    word_vetor_list = create_word_vector(path)
    sentence_vector_list = []
    for word_vetor_one in word_vetor_list:
        vetor_sum = np.zeros(model.vector_size)
        for word_vetor in word_vetor_one:
            try:
                vetor_sum += model.get_vector(word_vetor)
            except KeyError:
                pass
        sentence_vector_list.append(vetor_sum/len(word_vetor_one))
    return np.array(sentence_vector_list)
def Sentence_Kmeans(sentence_vector_list,k):
    Kmeans = KMeans(n_clusters=k)
    Kmeans.fit(sentence_vector_list)
    return Kmeans

def run(path,k):
    sentence_vector_list = sentence_vector(path)
    Kmeans = Sentence_Kmeans(sentence_vector_list, k)
    label_dic = defaultdict(list)
    for sent, label in zip(sentence_vector_list, Kmeans.labels_):
        label_dic[label].append(sent)
    label_dic1 = defaultdict(list)
    for label, li in label_dic.items():
        center_vector = Kmeans.cluster_centers_[label]
        for i in li:
            label_dic1[label].append(np.sqrt(np.sum((i - center_vector) ** 2)))
    for key, values in label_dic1.items():
        label_dic1[key] = np.mean(values)
    return sorted(label_dic1.items(), key=lambda x: x[1], reverse=False)

if __name__ == '__main__':
    path='titles1.txt'
    k=5
    dic_final = run(path=path, k=k)
   









