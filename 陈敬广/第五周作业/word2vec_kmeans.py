import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

'''
基于训练好的词向量模型进行聚类
聚类采用kmeans算法
'''


# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

# 加载标题文本,并返回每个标题文本分词后的集合
def load_sentences(path):
    # 集合的每个元素是每个标题文本分词 {'物业费 上涨 是否 需要 相关 部门 批准', '昆明 经适 房 遭弃   成 政府 烫手山芋',...}
    sentences = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.add(" ".join(jieba.cut(line.strip())))
    print("获取句子数量：", len(sentences))
    return sentences

# 将文本向量化
# sentences 文本分词后的数据，先将每个文本的所有分词映射成向量，再将所有词向量加权求和等手段映射层文本向量
def sentences_to_vectors(sentences,model):
    vectors = []
    for sentence in sentences:
        # 初始化文本（句子）向量，维度是词向量模型中的参数（因为是累加，所以词向量维度和句子向量维度是一样的）
        vector = np.zeros(model.vector_size)
        # 遍历每个句子的所有分词
        for word in sentence.split():
            try:
                # 将每个词向量进行累加(还可以拓展其他手段进行词向量到句子向量的映射)
                vector += model.wv[word]
            except KeyError:
                # 若句子分词后的某些词在词向量模型训练后不存在词向量中，则默认为全0向量
                vector += np.zeros(model.vector_size)
        # 变量完每个句子所有分词之后，得到所有分词的向量累加，求平均，映射为句子向量
        vectors.append(vector / len(sentence.split()))
    # 注意 np.array 和 python [] 区别
    return np.array(vectors)

# 计算两个向量的余弦值 vector1 * vector2 / ｜vector1｜*｜vector2｜
def cosine_similarity(vector1, vector2):
    x_dot_y = sum([x*y for x, y in zip(vector1, vector2)])
    sqrt_x = math.sqrt(sum([x ** 2 for x in vector1]))
    sqrt_y = math.sqrt(sum([x ** 2 for x in vector2]))
    if sqrt_y == 0 or sqrt_y == 0:
        return 0
    return x_dot_y / (sqrt_x * sqrt_y + 1e-7)

def main():
    # 加载训练好的词向量模型
    model = load_word2vec_model('model.w2v')
    # 加载所有待分类的标题，并返回每个标题文本分词后的集合
    sentences = load_sentences('titles.txt')
    # 将文本标题向量化
    vectors = sentences_to_vectors(sentences,model)

    # 指定聚类数目(人为指定)
    n_clusters = int(math.sqrt(len(sentences)))
    # 定义一个kmeans计算类
    kmeans = KMeans(n_clusters)
    # 聚类计算
    kmeans.fit(vectors)
    # 计算完成后所有分类标签，和文本句子一一对应，数量一致，总类是n_clusters
    labels = kmeans.labels_
    # 计算完成后所有分类标签的质心
    cluster_centers = kmeans.cluster_centers_

    # 标签_句子文本(分词后的)字典
    sentence_label_dict = defaultdict(list)
    for sentence,label in zip(sentences,labels):
        sentence_label_dict[label].append(sentence)
    # 标签_句子向量字典
    vector_label_dict = defaultdict(list)
    for vector,label in zip(vectors,labels):
        vector_label_dict[label].append(vector)

    label_cosine = dict()
    for label,vectors in vector_label_dict.items():
        # 标签值也是cluster_centers数组下标
        cluster_center = cluster_centers[label]

        # 遍历该标签下所有句子向量，并计算与质心向量的余弦相似度累加
        vector_center_cosine = 0.0
        for vector in vectors:
            vector_center_cosine += cosine_similarity(vector, cluster_center)
        # 标签下所有句子向量与质心向量的余弦相似度平均值，表示分类的凝聚度
        vector_center_cosine_avg = vector_center_cosine / len(vectors)
        # 记录到字典中
        label_cosine[label] = vector_center_cosine_avg
    # 遍历完成，将所有标签的余弦相似度平均值排序(reverse=True降序)，并返回列表数据结构
    label_cosine_desc = sorted(label_cosine.items(),key=lambda item:item[1],reverse=True)
    print(label_cosine_desc)
    for index,label_vector in enumerate(label_cosine_desc):
        if index == 3:
            return
        label = label_vector[0]
        print("cluster %s :" % label)
        sentences = sentence_label_dict.get(label)
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()