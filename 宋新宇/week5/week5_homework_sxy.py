import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial.distance import euclidean

"""
第5周作业：实现基于kmeans的类内距离计算，筛选优质类别。
"""

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
    model = load_word2vec_model("model_sxy.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    # sentence_label_dict = defaultdict(list)

    # 获取各个聚类的质心
    centers = kmeans.cluster_centers_
    # 初始化一个字典来存储每个类的平均距离
    cluster_info = {}
    # 计算每个类的平均距离
    for i in range(n_clusters):
        # 获取属于当前类的所有句子向量和句子
        cluster_vectors = [vectors[j] for j, label in enumerate(kmeans.labels_) if label == i]
        cluster_sentences = [list(sentences)[j] for j, label in enumerate(kmeans.labels_) if label == i]
        # 计算当前类所有句子到质心的距离，并计算平均值
        distances = [np.linalg.norm(vector - centers[i]) for vector in cluster_vectors]
        # distances = [euclidean(vector, centers[i]) for vector in cluster_vectors]
        average_distance = np.mean(distances)
        # 存储平均距离和句子列表
        cluster_info[i] = {'average_distance': average_distance, 'sentences': cluster_sentences}

        # 按平均距离从小到大排序
    sorted_cluster_info = sorted(cluster_info.items(), key=lambda x: x[1]['average_distance'])

    # 打印排序后的结果，包括每个类的平均距离和前10条句子
    num = int(input("请输入筛选出的优质聚类数量Top："))
    while num > 42:
        num = int(input("输入数量超过聚类总数，请重新输入Top："))
    n = 0
    for label, info in sorted_cluster_info:
        if n < num:
            n += 1
            print(f"Cluster {label}: 平均欧氏距离 = {info['average_distance']}")
            print("该类中代表句子（10条以内）:")
            for sentence in info['sentences'][:10]:  # 只打印前10条句子
                print(sentence.replace(" ", ""))
            # print("=================")
            # for i in range(min(10, len(info['sentences']))):
            #     print(info['sentences'][i].replace(" ", ""))
            print("\n")



    ###############

    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()
