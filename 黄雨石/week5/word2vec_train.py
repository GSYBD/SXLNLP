
import json
import jieba
import numpy as np
import gensim
from gensim.models import Word2Vec
from collections import defaultdict

'''
Word2Vec直接调用的这个直接实现
词向量模型的简单实现

'''

#训练模型
#corpus: [["cat", "say", "meow"], ["dog", "say", "woof"]]
#corpus: [["今天", "天气", "不错"], ["你", "好", "吗"]]
#dim指定词向量的维度，如100
def train_word2vec_model(corpus, dim):
    #调用Word2Vec直接训练完之后，sg指定训练方法，有默认，底层是写在c了'
    #参数windw 表示训练窗口的大小，min_count 过滤词频较低的，否者寻不好默认是5
    #sg 1 skipgram（中间预测两边） 0 CBOW（两边预测中间）
    #hs 使用霍夫曼树，默认做负采样
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
    #第一步读取一个预料文件
    with open("corpus.txt", encoding="utf8") as f:
        #读取每一行，遍历
        for line in f:
            #通过结巴分词，将一个句子，分词几个次放进一个列表中，将所有句子的分词结果，放进这里面
            sentences.append(jieba.lcut(line))
    #调用训练词向量的模型，传入所有的分词结果，好处是不需要标注和操作，只要做好分词预料的格式[["今天", "天气", "不错"], ["你", "好", "吗"]]
    #应该是一个数据了，结巴分词结果本身就是一个句子的分词结果，指定向量的维度，也就是需要训练的向量每个维度
    model = train_word2vec_model(sentences, 200)
    return model

if __name__ == "__main__":
    model = main()  #训练

    # model = load_word2vec_model("model.w2v")  #加载
    # #正向，表示接近的词， 女人就是表示不接近的词
    # print(model.wv.most_similar(positive=["男人", "母亲"], negative=["女人"])) #类比
    # #
    # while True:  #找相似
    #     string = input("input:")
    #     try:
    #         #model自带的功能，wv代表词向量代表找最接近的
    #         #学习是两个词的相关性，类似的语境出现，这是本质，
    #         #文本少，效果也不是很好 调整参数，也很会一档变化，但是因该不大 ，训练效果由训练数据决定，增加训练数据
    #         print(model.wv.most_similar(string))
    #     except KeyError:
    #         print("输入词不存在")