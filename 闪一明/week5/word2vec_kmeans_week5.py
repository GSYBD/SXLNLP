import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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

def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)  # 未出现的词用0向量
        vectors.append(vector / len(words))
    return np.array(vectors)

def main():
    model_path = r"F:\badouai\SXLNLP\闪一明\week5\model.w2v"
    data_path = "titles.txt"
    model = load_word2vec_model(model_path)
    sentences = load_sentence(data_path)
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    cluster_distances = defaultdict(list)
    for vector, sentence, label in zip(vectors, sentences, kmeans.labels_):
        distance = np.linalg.norm(vector - kmeans.cluster_centers_[label])
        cluster_distances[label].append((sentence, distance))

    # 对每个聚类中的句子按照距离排序并选取距离中心最近的10个句子
    sorted_clusters = sorted(cluster_distances.items())  # 按照聚类标签排序
    for label, data in sorted_clusters:
        data.sort(key=lambda x: x[1])  # 按照距离排序
        closest_sentences = data[:10]  # 取距离最近的10个
        print(f"Cluster {label}:")
        for sentence, dist in closest_sentences:
            print(sentence.replace(" ", ""), f"Distance: {dist:.2f}")
        print("--------")

if __name__ == "__main__":
    main()
