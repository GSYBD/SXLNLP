#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer
import json


"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()

        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        # 换装bert模型
        self.bert = BertModel.from_pretrained(r"..\..\week6 语言模型和预训练\bert-base-chinese", return_dict=False)

        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None , attention_mask = None):
        if y is not None:
            # 训练时，使用maskedattention
            
            # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
            # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
            # 换装bert模型, 传入maskedattention
            x, _= self.bert(x, attention_mask = attention_mask)  # output shape:(batch_size, sen_len, input_dim)

            y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)

            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1),ignore_index=-100)
        else:
            # 预测时，不用maskedattention
            x, _ = self.bert(x)  # output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
            return torch.softmax(y_pred, dim=-1)
    

#加载字表
def build_vocab(vocab_path):
    # 使用bert模型的字表
    vocab = tokenizer.vocab

    # vocab = {"<pad>":0}
    # with open(vocab_path, encoding="utf8") as f:
    #     for index, line in enumerate(f):
    #         char = line[:-1]       #去掉结尾换行符
    #         vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus =[]
    with open(path ,'r', encoding='utf-8') as f:
        # 遍历文件所有的行
        while True:
            data = f.readline()
            if not data:
                break
            # 判断文件是否空行
            if data.strip()!= "\n":
                corpus.append(data) 
    return corpus


#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(corpus,max_length):
    # start = random.randint(0, len(corpus) - 1 - window_size)
    # end = start + window_size
    # window = corpus[start:end]
    # target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    # y = [vocab.get(word, vocab["<UNK>"]) for word in target]

    # 生成一个小于corpus长度的随机数
    index = random.randint(0, len(corpus) - 1)
    json_sample = corpus[index]
    json_sample = json.loads(json_sample)
    x_sample = json_sample["title"]
    y_sample = json_sample["content"]
    
    # 使用bert的tokenizer 进行序列化
    x_sample = tokenizer.encode(x_sample, add_special_tokens=True, max_length=len(x_sample), padding="max_length", truncation=True)
    y_sample = tokenizer.encode(y_sample, add_special_tokens=True, max_length=max_length - len(x_sample), padding="max_length", truncation=True)
    y_sample = y_sample[1:] # 去除y_sample第一位的cls

    xx_mask = torch.ones((len(x_sample),len(x_sample)))
    xy_mask = torch.zeros((len(x_sample),len(y_sample))) 
    yx_mask = torch.ones((len(y_sample),len(x_sample)))
    yy_mask = torch.tril(torch.ones((len(y_sample),len(y_sample))))

    # 拼接mask
    # 先对x、y分别进行横向拼接
    x_mask = torch.cat((xx_mask,xy_mask),dim=1)
    y_mask = torch.cat((yx_mask,yy_mask),dim=1)
    # 再对x、y进行纵向拼接
    mask = torch.cat((x_mask,y_mask),dim=0)

    # 构造x_sample，y_sample
    x_sample_t = torch.tensor(x_sample)
    x_sample = x_sample + y_sample

    # 构造loss函数带ignore_index值的真值矩阵,-100表示忽略的位置，只计算句子的后半部分
    # 设置x_mask_t的值为-100
    x_sample_t = x_sample_t.fill_(-100)
    # 构造y_sample，y_sample进行横向拼接
    y_sample = x_sample_t[:-1].tolist() + y_sample +[-100] # label 要错开一位
 
    return x_sample,y_sample ,mask

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#x_max_length, y_max_length 样本长度
#corpus 语料字符串
def build_dataset(sample_length, max_length, corpus):
    dataset_x = []
    dataset_y = []
    dataset_mask =None
    for i in range(sample_length):
        x,y, mask = build_sample( corpus,max_length)
        dataset_x.append(x)
        dataset_y.append(y)
        # 对mask进行纵向拼接
        dataset_mask = torch.cat((dataset_mask,mask.unsqueeze(0)),dim=0) if dataset_mask is not None else mask.unsqueeze(0)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), dataset_mask

#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 130:
            openings += pred_char
            # x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            # 使用bert的tokenizer 进行序列化
            x = tokenizer.encode(openings[-window_size:], add_special_tokens=True, max_length=window_size, padding="max_length", truncation=True)

            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            # pred_char = reverse_vocab[index]
            # 使用bert的tokenizer进行解码
            pred_char = tokenizer.decode([index], skip_special_tokens=True)
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]

            # x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            # 使用bert的tokenizer 进行序列化
            x = tokenizer.encode(window, add_special_tokens=True, max_length=window_size, padding="max_length", truncation=True)
            
            x = torch.LongTensor([x])
            target = sentence[i]

            # 使用bert的tokenizer的词表获取索引
            target_index = tokenizer.convert_tokens_to_ids(target)
            # target_index = vocab.get(target, vocab["<UNK>"])
            # print(target_index)

            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 20000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 100       #样本文本长度
    vocab = build_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, mask = build_dataset(batch_size, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("城会玩都是这么健身的 一言不合就射一箭", model, vocab, window_size))
        print(generate_sentence("新西兰奶粉全年进口配额1月耗光 大旱或推涨国内售价", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

tokenizer = BertTokenizer.from_pretrained(r"..\..\week6 语言模型和预训练\bert-base-chinese", do_lower_case=True)

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"..\..\week10 文本生成问题\transformers-生成文章标题\sample_data.json", False)
