import jieba
import math
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

"""
语料库没有换, titles也用的是上课的素材
根据语料库以及维度训练词向量模型
加载训练好的词向量模型
利用词向量模型将句子转化为向量
对用户输入的句子进行聚类计算
"""

class W2VModel:
    def __init__(self, corpus_path: str, dim: int) -> None:
        self.corpus_path = str(Path(__file__).parent / corpus_path)
        self.model_path = str(Path(__file__).parent / "model.w2v")
        self.dim = dim
    
    def train_model(self, sentences):
        model = Word2Vec(sentences, vector_size=self.dim, sg=1)
        model.save(self.model_path)
        return model
    
    def mk_model(self):
        sentences = []
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                sentences.append(jieba.lcut(line.strip()))
                    
        model = self.train_model(sentences)
        return model
    
    def load(self):
        model = Word2Vec.load(self.model_path) if self.model_path is not None else None
        return model
        
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            if word in model.wv:
                vector += model.wv[word]
        vectors.append(vector / len(words) if words else vector)
    return np.array(vectors)

def load_sentence(path):
    sentences = []
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.append(" ".join(jieba.cut(sentence)))
    print("获得的句子数量：", len(sentences))
    return sentences

def main():
    # Implement a model by the corpus data and the specialized dimension
    model_obj = W2VModel("corpus.txt", 100)
    # Train and save the model named model.w2v
    model_obj.mk_model()
    print(f"model_path: {model_obj.model_path}")
    
    model = model_obj.load()
    title_path = Path(__file__).parent / "titles.txt"
    
    sentences = load_sentence(str(title_path))
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(vectors)
    
    label_dict = defaultdict(list)
    for s, l in zip(sentences, kmeans.labels_):
        label_dict[l].append(s)
    print(label_dict)
    for l, s in label_dict.items():
        print(f"cluster :: {l}, s:: {len(s)}")
        for i in range(min(10, len(s))):
            print(s[i].replace(" ", ""))
        print("-" * 50)
        
if __name__ == "__main__":
    main()
