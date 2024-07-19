import jieba
from gensim.models import Word2Vec

# 训练词向量模型
def train_word2vec_model(corpus,dim):
    model=Word2Vec(corpus,vector_size=dim,sg=1) # sg=1:skip-gram,otherwise:cbow
    model.save("model.w2v")
    print("词向量训练完成！")

# 构造语料
def construct_corpus(corpus_path):
    corpus_list=[]
    with open(corpus_path,encoding='utf-8') as f:
        for line in f:
            corpus_list.append(jieba.lcut(line))
    return corpus_list

if __name__=="__main__":
    corpus_path="corpus.txt"
    dim=100
    corpus=construct_corpus(corpus_path)
    train_word2vec_model(corpus,dim)