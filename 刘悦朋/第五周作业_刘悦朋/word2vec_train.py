
import json
import jieba
import numpy as np
import gensim
from gensim.models import Word2Vec
from collections import defaultdict

'''
词向量模型的简单实现
'''

#训练模型
#corpus: [["cat", "say", "meow"], ["dog", "say", "woof"]]
#corpus: [["今天", "天气", "不错"], ["你", "好", "吗"]]
#dim指定词向量的维度，如100
def train_word2vec_model(corpus, dim):
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    model.save("model.w2v")
    return model

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def main():
    sentences = []
    with open("corpus.txt", encoding="utf8") as f:
        for line in f:
            sentences.append(jieba.lcut(line))
    model = train_word2vec_model(sentences, 100)
    return model

if __name__ == "__main__":
    # model = main()  #训练

    model = load_word2vec_model("model.w2v")  #加载
 
    print(model.wv.most_similar(positive=["男人", "母亲"], negative=["女人"])) #类比

    while True:  #找相似
        string = input("input:")
        try:
            print(model.wv.most_similar(string))
        except KeyError:
            print("输入词不存在")