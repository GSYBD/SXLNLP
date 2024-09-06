#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertModel, BertTokenizer

"""
基于pytorch的bert语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.layer = BertModel.from_pretrained(r"E:\badouai\ai\week6\bert-base-chinese", return_dict = False)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask_tensor=None):
        if y is not None:
            x, _ = self.layer(x, attention_mask = mask_tensor)        #output shape:(batch_size, sen_len, input_dim)
        else:
            x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index = -100)
        else:
            return torch.softmax(y_pred, dim=-1)



#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index
    return vocab

#加载语料
def load_corpus(path):
    corpus_title = []
    corpus_content = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line['title']
            content = line['content']
            corpus_title.append(title.strip())
            corpus_content.append(content.strip())
    return corpus_title, corpus_content

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, sentence_max_length, corpus_q, corpus_a):
    index = random.randint(0, len(corpus_q) - 1)
    corpus_window = corpus_q[index]
    corpus_target = corpus_a[index]
    start_q = random.randint(0, max(len(corpus_window) - 1 - window_size , 1))
    end_q = start_q + window_size
    window = corpus_window[start_q:end_q]
    start_a = random.randint(0, max(len(corpus_target) - sentence_max_length + window_size, 1))
    end_a = start_a + sentence_max_length - window_size -1
    target = corpus_target[start_a:end_a]
    mask_tensor = build_mask_tensor(window_size, sentence_max_length)
    ####### ！！！！！修改输入和输出的长度，保证输入的长度为sentence_max_length，输出的长度为sentence_max_length - window_size

    # print(window, target)
    Tokenizer = BertTokenizer.from_pretrained(r"E:\badouai\ai\week6\bert-base-chinese")
    x = Tokenizer.encode(window, add_special_tokens=False, truncation=True, max_length=window_size, padding = 'max_length')
    y = Tokenizer.encode(target, add_special_tokens=False, truncation=True, max_length=sentence_max_length - window_size, padding = 'max_length')
    if len(x) < window_size:
        x = x + [vocab["[PAD]"]] * (window_size - len(x))
    for i in range(window_size):
        y.insert(0, -100)
    if len(y) < sentence_max_length:
        y = y + [vocab["[PAD]"]] * (sentence_max_length - len(y))
    x = x + [vocab["[SEP]"]] + y[10:-1]
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    # y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y, mask_tensor

def build_mask_tensor(window_size, sentence_max_length):
    mask_tensor = torch.zeros((sentence_max_length, sentence_max_length))      ##### 有待修改！！！！！！！！！，mask_tensor的形状应该是（batch, sentence_max_length, sentence_max_length）
    for i in range(sentence_max_length):
        for j in range(sentence_max_length):
            if j < window_size or i >= j:
                mask_tensor[i][j] = 1
            else:
                mask_tensor[i][j] = 0
    return mask_tensor

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, sentence_max_length, corpus_q, corpus_a):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y, mask_tensor_one_batch = build_sample(vocab, window_size, sentence_max_length, corpus_q, corpus_a)
        dataset_x.append(x)
        dataset_y.append(y)
        mask_tensor_batch = mask_tensor_one_batch.unsqueeze(0)
        if i < 1:
            mask_tensor = mask_tensor_batch
        else:
            mask_tensor = torch.cat((mask_tensor, mask_tensor_batch), dim=0)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), mask_tensor

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
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            Tokenizer = BertTokenizer.from_pretrained(r"E:\badouai\ai\week6\bert-base-chinese")
            x = Tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
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
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #question的长度
    sentence_max_length = 50    # 总文本长度的最大值
    vocab = build_vocab(r"E:\badouai\ai\week6\bert-base-chinese\vocab.txt")       #建立字表
    corpus_q, corpus_a = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, mask_tensor = build_dataset(batch_size, vocab, window_size, sentence_max_length, corpus_q, corpus_a) #构建一组训练样本
            if torch.cuda.is_available():
                x, y, mask_tensor = x.cuda(), y.cuda(), mask_tensor.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask_tensor)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        # print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"E:\badouai\ai\第十周 生成式任务\week10 文本生成问题\transformers-生成文章标题\sample_data.json", False)
