import json
import math
import jieba
import numpy as np
import gensim
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.cluster import KMeans
"""
词向量的简单实现
"""
#加载词向量模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model
#加载句子库
def load_sentence(path):
    sentences= []
    with open(path,'r',encoding='utf8') as f:
        for line in f:
            sentence = line.strip()
            sentences.append(' '.join(jieba.cut(sentence)))
    print('获取句子条目：%d'%len(sentences))
    return sentences #输出句子长度
#建立文本向量

def sentence_to_vector(sentences,model):
    vectors = []
    for line in sentences:
        words = line.strip()
        vector = np.zeros(model.vector_size) #获取模型的向量尺寸
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector/len(line))
    return np.array(vectors)


            
def main():
    model_path = 'model.w2v'
    path = 'titles.txt'
    model = load_word2vec_model(model_path)
    sentences = load_sentence(path)
    vectors = sentence_to_vector(sentences,model)
    n_clusters = int(math.sqrt(len(sentences)))
    print('指定聚类数目：%d'%n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)
    
    sentence_label_dict = defaultdict(list)
    for sentence,label in zip(sentences,kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    
    # print(sentence_label_dict[1][1].replace(' ', ''))
    for label,sentence in sentence_label_dict.items():
        print("cluster %s :"%label)
        for i in range(min(10,len(sentence))):
            print(sentence[i].replace(' ',''))

if __name__ =='__main__':
    main()