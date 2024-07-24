import math
import re
import json
import jieba
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from collections import OrderedDict


###训练WordVec2 模型 dim = 50 saved in model.w2x ##

#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

##加载需要进行聚类的句子并做jieba切分
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

#####根据类内平均距离进行排序

def cluster_sort(sentence_label,model):
    avg_dis = [0]*len(sentence_label.keys())
    for i in sentence_label.keys():
        sentences_list = sentence_label.get(i) #找出每个类别中的句子
        clust = sentences_to_vectors(sentences_list,model) #句子转换成向量
        clust_mean = clust.mean(axis=0) #类质心
        avg_dis[i] = np.sqrt(np.sum(np.square(clust - clust_mean))) #欧式距离

    order = np.argsort(avg_dis) # 对距离从小到大排序 记下index -- 重新排序的顺序
    sorted_dict = defaultdict(list)
    for j in order:
        key = (j, avg_dis[j])
        sorted_dict[key] = sentence_label[j]
    return sorted_dict



def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    sorted_sentence_label_dict = cluster_sort(sentence_label_dict,model) #对cluster排序 - 保留原cluster的序号以及额外输出类内距离

    for label, sentences in sorted_sentence_label_dict.items():
        print("cluster %d within dist %.2f:" % (label))
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()

