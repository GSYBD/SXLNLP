import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


#加载预料并分词
def loadfile(path):
    sentence = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line.strip()
            sentence.append(" ".join(jieba.lcut(line)))
    return sentence

#语句序列化
def Tosequence(sentence, model):
    vectors = []
    for line in sentence:
        vector = np.zeros(model.vector_size)
        for word in line:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector/len(line))
    # print("vectors", vectors)
    # print("array vectors", np.array(vectors))
    return np.array(vectors)


def main():
    model = Word2Vec.load("model.w2v")
    sentences = loadfile("titles.txt")
    vectors = Tosequence(sentences, model)
    cluster = int(np.sqrt(len(sentences)))
    kmeans = KMeans(cluster)
    kmeans.fit(vectors)

    kmeans_dic = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        kmeans_dic[label].append(sentence)

    kmeans_dic_distance = defaultdict(dict)
    for i in range(0, cluster):
        distance = 0
        point = kmeans.cluster_centers_[i] #获取每个类别的中心点
        sequeces = Tosequence(kmeans_dic[i], model) #每个类别的语句序列化
        for j in range(0, len(sequeces)):
            distance += np.linalg.norm(np.array(sequeces[j]) - np.array(point))  #每个类别中的向量和中心点的距离和
        distance_avg = distance/len(sequeces) #求平均距离
        kmeans_dic_distance[i] = distance_avg
    sorted_data = sorted(kmeans_dic_distance.items(), key=lambda x:x[1])  #对类别的平均距离排序
    top10 = dict(sorted_data[:10]).keys() #拿到距离最小的前10个类别


    print("最优质的10个种类为：")
    for label in top10:
        sentences = kmeans_dic[label]
        print("cluster:", label)
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""), end="")
        print("---------------------------")



if __name__ == "__main__":
    main()


