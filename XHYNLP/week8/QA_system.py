import os
import json
import jieba
import numpy as np
from bm25 import BM25
from similarity_function import editing_distance, jaccard_distance
from gensim.models import Word2Vec

'''
基于faq知识库和文本匹配算法进行意图识别，完成单轮问答
'''

class QASystem:
    def __init__(self, know_base_path, algo):
        '''
        :param know_base_path: 知识库文件路径
        :param algo: 选择不同的算法
        '''
        self.load_know_base(know_base_path)
        self.algo = algo
        if algo == "bm25":
            self.load_bm25()
        elif algo == "word2vec":
            self.load_word2vec()
        else:
            #其余的算法不需要做事先计算
            pass

    def load_bm25(self):
        self.corpus = {}
        for target, questions in self.target_to_questions.items():
            self.corpus[target] = []
            for question in questions:
                self.corpus[target] += jieba.lcut(question)
        self.bm25_model = BM25(self.corpus)

    #词向量的训练
    def load_word2vec(self):
        #词向量的训练需要一定时间，如果之前训练过，我们就直接读取训练好的模型
        #注意如果数据集更换了，应当重新训练
        #当然，也可以收集一份大量的通用的语料，训练一个通用词向量模型。一般少量数据来训练效果不会太理想
        if os.path.isfile("model.w2v"):
            self.w2v_model = Word2Vec.load("model.w2v")
        else:
            #训练语料的准备，把所有问题分词后连在一起
            corpus = []
            for questions in self.target_to_questions.values():
                for question in questions:
                    corpus.append(jieba.lcut(question))
            #调用第三方库训练模型
            self.w2v_model = Word2Vec(corpus, vector_size=100, min_count=1)
            #保存模型
            self.w2v_model.save("model.w2v")
        #借助词向量模型，将知识库中的问题向量化
        self.target_to_vectors = {}
        for target, questions in self.target_to_questions.items():
            vectors = []
            for question in questions:
                vectors.append(self.sentence_to_vec(question))
            self.target_to_vectors[target] = np.array(vectors)

    # 将文本向量化
    def sentence_to_vec(self, sentence):
        vector = np.zeros(self.w2v_model.vector_size)
        words = jieba.lcut(sentence)
        # 所有词的向量相加求平均，作为句子向量
        count = 0
        for word in words:
            if word in self.w2v_model.wv:
                count += 1
                vector += self.w2v_model.wv[word]
        vector = np.array(vector) / count
        #文本向量做l2归一化，方便计算cos距离
        vector = vector / np.sqrt(np.sum(np.square(vector)))
        return vector

    def load_know_base(self, know_base_path):
        self.target_to_questions = {}
        with open(know_base_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]
                target = content["target"]
                self.target_to_questions[target] = questions
        return

    def query(self, user_query):
        results = []
        if self.algo == "editing_distance":
            for target, questions in self.target_to_questions.items():
                scores = [editing_distance(question, user_query) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "jaccard_distance":
            for target, questions in self.target_to_questions.items():
                scores = [jaccard_distance(question, user_query) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "bm25":
            words = jieba.lcut(user_query)
            results = self.bm25_model.get_scores(words)
        elif self.algo == "word2vec":
            query_vector = self.sentence_to_vec(user_query)
            for target, vectors in self.target_to_vectors.items():
                cos = query_vector.dot(vectors.transpose())
                print(cos)
                results.append([target, np.mean(cos)])
        else:
            assert "unknown algorithm!!"
        sort_results = sorted(results, key=lambda x:x[1], reverse=True)
        return sort_results[:3]


if __name__ == '__main__':
    qas = QASystem("data/train.json", "bm25")
    question = "我想重置下固话密码"
    res = qas.query(question)
    print(res)
    #
    # while True:
    #     question = input("请输入问题：")
    #     res = qas.query(question)
    #     print("命中问题：", res)
    #     print("-----------")
