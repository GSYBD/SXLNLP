#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

"""
基于pytorch的LSTM语言模型
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy
import json
import random
class positionalencoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(positionalencoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        # for pos in range(max_seq_len):
        #     for i in range(0, d_model, 2):
        #         pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
        #         pe[pos, i+1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x * math.sqrt(self.d_model)
        # print(x.size())
        # print(self.pe.size())
        x = x + self.pe[:, :x.size(1)]
        return x

class encoderlayer(nn.Module):
    def __init__(self, d_model, head, dropout = 0.1):
        super().__init__()
        self.norm1 = normallayer(d_model)
        self.norm2 = normallayer(d_model)
        self.attn = mutihead_attention(d_model, head)
        self.ffn = ffn(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        attn = self.attn(x2, x2, x2, mask)
        x = x2 + self.dropout_1(attn)
        x2 = self.norm2(x)
        x = x + self.dropout_2(self.ffn(x2))
        return x

class normallayer(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        norm = self.gamma * (x - x.mean(dim = -1, keepdim = True)) \
        / (x.std(dim = -1, keepdim = True) + self.eps) + self.beta
        return norm

class ffn(nn.Module):
    def __init__(self, d_model, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class mutihead_attention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.o_linear = nn.Linear(d_model, d_model)

    def attention_(self, q, k, v, mask=None, droup=None):
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if droup is not None:
            scores = droup(scores)
        output = torch.matmul(scores, v)
        return output

    def group_attenstion(self, q, k, v, groupsize):
        t_seq = q.shape[2]
        n_seq = t_seq // groupsize
        att_output = []
        for i in range(n_seq):
            q_tmp = q[:, :, groupsize * i: groupsize * (i + 1), :]
            k_tmp = k[:, :, groupsize * i: groupsize * (i + 1), :]
            v_tmp = v[:, :, groupsize * i: groupsize * (i + 1), :]
            scores_tmp = torch.matmul(q_tmp, k_tmp.transpose(-1, -2))
            scors_tmp = scores_tmp / math.sqrt(self.head_dim)
            scors_tmp = F.softmax(scors_tmp, dim=-1)
            output_tmp = torch.matmul(scors_tmp, v_tmp)
            att_output.append(output_tmp)

        att_output = torch.cat(att_output, dim=2)
        return att_output


    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q = q.view(bs, -1, self.heads, self.head_dim)
        k = k.view(bs, -1, self.heads, self.head_dim)
        v = v.view(bs, -1, self.heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = self.attention_(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, -1, self.heads * self.head_dim)
        output = self.o_linear(output)
        return output

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_dim)
        self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
        # print(y_pred.shape)
        # print(y.shape)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, head, dropout=0.1):
        super().__init__()
        self.N = N
        self.vocab_size = len(vocab_size)
        self.embed = nn.Embedding(len(vocab_size), d_model)
        self.pe = positionalencoder(d_model)
        self.classify = nn.Linear(d_model, len(vocab_size))
        self.en_layers = nn.ModuleList([copy.deepcopy(encoderlayer(d_model, head, dropout)) for _ in range(N)])
        self.normallayer = normallayer(d_model)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, src, y=None, mask=None):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.en_layers[i](x, mask)
        x = self.normallayer(x)
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=1)
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串

#建立模型
def build_model(vocab, char_dim):
    # model = LanguageModel(char_dim, vocab)
    model = Encoder(vocab, char_dim, 12, 12)
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
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
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

def load_jsons(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            corpus.append(title + "||" + content)
    return corpus

def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def create_prefix_mask(x_win, y_win, seq_len):
    # 获取 x_win 和 y_win 的长度
    x_len = len(x_win)
    y_len = len(y_win)

    # 创建一个大小为 seq_len 的掩码
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    # x_win 部分是相互可见的
    mask[:x_len, :x_len] = True

    # y_win 部分是斜着的三角矩阵
    for i in range(y_len):
        mask[x_len + i, :x_len + i + 1] = True

    # 最后的 pad 部分是完全不可见的
    pad_len = seq_len - x_len - y_len
    mask[x_len + y_len:, :] = False

    return mask

def build_dataset2(batch_size, vocab, corpus, seq_len = 512):
    dataset_x = []
    dataset_y = []

    dataset_mask = []
    for i in range(batch_size):
        tmp = random.choice(corpus).split("||")

        x_win = tmp[0]
        y_win = tmp[1]
        inputk = x_win + y_win

        if len(inputk) > seq_len:
            inputk = inputk[:seq_len]
        target = x_win + y_win
        target = target[1:]
        if len(target) > seq_len:
            target = target[:seq_len]

        x = [vocab.get(word, vocab['[UNK]']) for word in inputk]   #将字转换成序号
        y = [vocab.get(word, vocab['[UNK]']) for word in target]
        y.append(vocab['[SEP]'])

        if len(x) < seq_len:
            x = x + [vocab['[PAD]']] * (seq_len - len(x))
        if len(y) < seq_len:
            y = y + [vocab['[PAD]']] * (seq_len - len(y))

        y[:len(x_win)-1] = [1] * (len(x_win) - 1)
        # print(x)
        # print(y)
        mask = create_prefix_mask(x_win, y_win, seq_len)

        dataset_x.append(x)
        dataset_y.append(y)
        dataset_mask.append(mask.unsqueeze(0))

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), torch.cat(dataset_mask, dim=0)

def generate_sentence2(openings, model, vocab):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != '[PAD]' and len(openings) <= 20 and pred_char != '[SEP]':
            openings += pred_char
            x = [vocab.get(char, vocab["[UNK]"]) for char in openings]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
            print(pred_char)
    return openings

def train(corpus_path, save_weight=False):
    epoch_num = 20        #训练轮数
    batch_size = 2       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度
    vocab = build_vocab(r"D:\pythonfile\华中杯\八斗AI\week10 文本生成问题\transformers-生成文章标题\vocab.txt")       #建立字表
    # print(vocab)
    corpus = load_jsons(corpus_path)     #加载语料
    # print(corpus)
    model = build_model(vocab, char_dim)    #建立模型
    # src_mask = torch.tril(torch.ones(window_size, window_size)).bool()
    # # print(model)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    # print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, mask = build_dataset2(batch_size, vocab, corpus, seq_len = 512) #构建一组训练样本

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            print(loss.item())
            print(generate_sentence2("王金平关说案：窃听风暴引发政坛地震", model, vocab))
            break
        break
            # loss.backward()      #计算梯度
            # optim.step()         #更新权重
            # watch_loss.append(loss.item())
        # print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

    # if not save_weight:
    #     return
    # else:
    #     base_name = os.path.basename(corpus_path).replace("txt", "pth")
    #     model_path = os.path.join("model", base_name)
    #     torch.save(model.state_dict(), model_path)
    #     return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"D:\pythonfile\华中杯\八斗AI\week10 文本生成问题\transformers-生成文章标题\sample_data.json", False)
