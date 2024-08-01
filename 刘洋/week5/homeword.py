#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
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


def main():
    model = load_word2vec_model(r"D:\刘洋\刘洋(个人)\2024大数据算法学习\学习资料\第五周 词向量\week5 词向量及文本向量\model.w2v")  #加载词向量模型
    sentences = load_sentence(r"D:\刘洋\刘洋(个人)\2024大数据算法学习\学习资料\第五周 词向量\week5 词向量及文本向量\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # 计算聚类中心
    center = kmeans.cluster_centers_
    sentence_label_dict = defaultdict(list)
    cluster_distance = defaultdict(list)

    # 将句子和标签对应起来
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):
        sentence_label_dict[label].append(sentence)
        distance = np.linalg.norm(vector - center[label])
        cluster_distance[label].append(distance)

    #计算平均距离
    mean_distance = {}
    for label, distances in cluster_distance.items():
        # 计算当前簇的平均距离
        mean_distance[label] = np.mean(distances)

    #根据平均距离进行排序
    new_label = sorted(mean_distance.keys(), key=lambda label:mean_distance[label])

    for label in new_label:
        print("cluster %s --> distance_mean: %f:" % (label,mean_distance[label]))
        for i in range(min(10, len(sentence_label_dict[label]))):  # 随便打印几个，太多了看不过来
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()