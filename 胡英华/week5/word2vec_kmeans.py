import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

"""基于训练好的词向量模型进行KMeans聚类"""


"""
model.wv 是 Gensim Word2Vec 模型中的一个属性，它表示模型的词向量（word vectors）。
在 Gensim 的 Word2Vec 模型中，每个词都有一个对应的词向量。
词向量是一个多维的实数向量，用于表示词的语义信息。通过计算词向量的相似度，可以衡量词与词之间的语义关系。
model.wv 属性是一个 KeyedVectors 对象，它提供了一些方法来访问和操作词向量，例如：
model.wv['word']：获取词的词向量。
model.wv.most_similar('word')：获取与给定词最相似的词。
model.wv.similarity('word1', 'word2')：计算两个词的相似度。
通过 model.wv 属性，可以方便地使用 Word2Vec 模型进行词向量的计算和词义相似度的分析。


kmeans.fit(vectors) 是在使用 scikit-learn 库中的 KMeans 聚类算法对数据进行聚类。
1.kmeans：这是一个 KMeans 聚类器对象，它已经通过 KMeans(n_clusters=n) 创建，其中 n 是聚类的簇数。
2.fit(vectors)：这是 KMeans 聚类器对象的一个方法，用于对输入的数据 vectors 进行聚类。vectors 是一个二维数组，每一行代表一个样本，每一列代表一个特征。
3.fit 方法会根据输入的数据 vectors 计算出每个样本所属的簇，并将结果保存在 kmeans.labels_ 属性中。同时，fit 方法还会计算出每个簇的中心点，保存在 kmeans.cluster_centers_ 属性中。

sentence_label_dict = defaultdict(list) 是 Python 中的一个语句，它使用了 collections 模块中的 defaultdict 类。
defaultdict 是 Python 的内置数据类型 dict 的一个子类，它接受一个函数作为参数，并使用这个函数来生成字典中不存在的键的默认值。
在这个例子中，defaultdict(list) 创建了一个 defaultdict 对象，它使用 list 函数来生成默认值。这意味着，当尝试访问 sentence_label_dict 中不存在的键时，defaultdict 会自动创建一个空列表作为该键的值。

sentence = set() 是 Python 中的一个语句，它创建了一个空集合（set）。
在 Python 中，集合是一种无序且不重复的元素集。集合中的元素必须是不可变类型，例如整数、浮点数、字符串、元组等，而不能是列表、字典或其他集合。
集合有一些常用的操作，例如：

添加元素：sentence.add('word')，将 'word' 添加到集合中。
删除元素：sentence.remove('word')，从集合中删除 'word'。
检查元素是否存在：'word' in sentence，如果 'word' 在集合中，返回 True，否则返回 False。
集合的元素是无序的，因此不能通过索引访问集合中的元素。但是，可以通过遍历集合来访问其中的元素。

需要注意的是，集合中的元素必须是唯一的，不能有重复的元素。如果尝试向集合中添加一个已经存在的元素，集合不会发生改变。

"""


# 加载训练好的模型
def load_word2vec_model(model_path):
    model = Word2Vec.load(model_path)
    return model

 
# 加载句子数据
def load_sentence(path):
    # sentence = set() 是 Python 中的一个语句，它创建了一个空集合（set）。
    sentences = set()
    with open(path, encoding="utf8") as f:
        # 拿第一行来举例
        for line in f:  # line : 新增资金入场 沪胶强势创年内新高
            # sentence : 新增资金入场 沪胶强势创年内新高
            sentence = line.strip()  # 删除字符串中多余的空格和特殊字符  取出每一行的句子
            # {'新增 资金 入场   沪 胶 强势 创 年内 新高'}
            # {'NYMEX 原油期货 电子盘 持稳 在 75 美元 下方', '新增 资金 入场   沪 胶 强势 创 年内 新高'}
            sentences.add(" ".join(jieba.cut(sentence))) # 
    print("获取句子数量: ", len(sentences))
    return sentences  # 返回分好词的句子

            
# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        # print(words) #['叶剑英', '次', '子叶', '选宁', '少将', '去世', '母亲', '为', '曾国藩', '后裔'] 
        # input()
        vector = np.zeros(model.vector_size) # model.vector_size = 100
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector)
    return np.array(vectors)  # 返回句子向量列表
                
        
# def main():
#     model = load_word2vec_model("model.w2v")
#     sentences = load_sentence("titles.txt")  # 加载所有标题
#     vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化
#     n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量  自定义
#     print("指定聚类数量: ", n_clusters)
#     kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
#     kmeans.fit(vectors)  # 进行聚类计算
#     sentence_label_dict = defaultdict(list)  # 定义一个字典，用于存储句子与其对应的标签

#     for sentence, label in zip(sentences, kmeans.labels_):  # 遍历句子和其对应的标签
#         sentence_label_dict[label].append(sentence)  # 将句子和其对应的标签存入字典中      
#     for label, sentences in sentence_label_dict.items():  # 遍历字典，打印每个标签对应的句子
#         print("cluster: %s" % label)
#         for i in range(min(10, len(sentences))):
#             print(sentences[i].replace(" ", ""))
#         print("------------")


# 计算余弦距离
def cosine_distance(vector1, vector2):
    # 计算点积
    dot_product = np.dot(vector1, vector2)
    # 计算模长
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    # 计算余弦距离
    cosine_distance = dot_product / (norm1 * norm2)
    return 1 - cosine_distance


def main():
    model = load_word2vec_model(r"./model.w2v")
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量  自定义
    print("指定聚类数量: ", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算
    sentence_label_dict = defaultdict(list)  # 定义一个字典，用于存储句子与其对应的标签


    # 保存每个句子的标签
    distance_label_dict = defaultdict(list)
    distance_dict = defaultdict(list)
    center = kmeans.cluster_centers_  # 获取聚类中心
    for index, label in enumerate(kmeans.labels_):
        vector = vectors[index]
        distance = cosine_distance(vector,center[label])  # 计算与聚类中心的距离
        distance_label_dict[label].append(distance)
    for label, distance_list in distance_label_dict.items():
        distance_dict[label].append(np.mean(distance_list))  # 计算每个聚类中心的平均距离
    # key 参数指定一个函数，用于从每个元素中提取比较键。这里使用了一个匿名函数 lambda x: x[1]，它表示从每个键值对 (key, value) 中提取 value 作为比较的依据。  
    # reverse=True，则列表元素将按照降序排序
    distance_order = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)  # 按照平均距离排序 由大到小
    # 输出前
    for label,cos_tance in distance_order[:10]:
        print(f"label:{label} , distance: {cos_tance}")




"""Driver Code"""
if __name__ == '__main__':
    main()













