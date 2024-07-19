import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 输入模型文件路径
# 加载训练好的模型
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

# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    center = kmeans.cluster_centers_  # 每个分类的中心点向量
    sentence_distance_dict = defaultdict(list)  # 创建dict存储距离和标签
    sentence_dl_dict = defaultdict(list)    #创建dict存储舍弃后的句子

    for label, sentences in sentence_label_dict.items():
        vectors=[]
        for sentence in sentences:
            words = sentence.split()  # sentence是分好词的，空格分开
            vector = np.zeros(model.vector_size)
            # 所有词的向量相加求平均，作为句子向量
            for word in words:
                try:
                    vector += model.wv[word]
                except KeyError:
                    # 部分词在训练中未出现，用全0向量代替
                    vector += np.zeros(model.vector_size)
            vectors.append(vector / len(words))
            distance = np.linalg.norm(center[label] - vectors)  # 计算每个句子和中心的距离
            sentence_distance_dict[label].append((distance, sentence))  # 将标签,(句子.距离)归类

    for label, distance in sentence_distance_dict.items():
        sort_dis = sorted(distance, key=lambda x: x[0])     #对距离排序
        top20 = [s for d, s in sort_dis[:20]]     #只要前20
        sentence_dl_dict[label].append(top20)

    for label, sentences in sentence_dl_dict.items():
        print(np.array(sentences).shape)
        sentences=np.array(sentences)
        sentences=np.squeeze(sentences)
        print(sentences.shape)
        print("clustr %s :" % label)
        result = ""
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            result=sentences[i].replace(" ", "")
            print(result)
        print("---------")

if __name__ == "__main__":
    main()