# coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel
from transformers import BertTokenizer
"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)  # 设置 ignore_index

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, mask, x, y=None):
        # x = self.embedding(x)  # output shape:(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x)  # output shape:(batch_size, sen_len, input_dim)
        # 生成下三角矩阵 (causal mask)
        # seq_len = x.size(1)
        # batch_size = x.size(0)
        # causal_mask = torch.tril(torch.ones(seq_len, seq_len))  # 生成下三角矩阵
        # causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展维度
        if torch.cuda.is_available():
            mask = mask.cuda()
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    if "<UNK>" not in vocab:
        vocab["<UNK>"] = len(vocab)
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 处理数据
def load_title_content(path):
    corpus = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 确保当前行不是空的
                item = json.loads(line.strip())  # 解析当前行的JSON
                title = item.get('title', '')
                content = item.get('content', '')
                combined_string = title + "+" + content
                corpus.append(combined_string)
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, corpus):
    index = random.randint(0, len(corpus) - 1)
    combined_string = corpus[index]
    title, content = combined_string.split("+", 1)
    t_len, c_len = len(title), len(content)
    # 定义一个特殊的分隔符token，这里可以使用[SEP]或者你自定义的符号
    separator_token = "[SEP]"
    end_token = "[CLS]"  # 使用 [CLS] 作为结束符
    # 将title和content连接起来，中间用分隔符隔开
    combined_text_x = title + content
    combined_text_y = title + content
    # print(window, target)
    # 在这生成mask
    total_length = t_len + c_len
    mask = np.zeros((200, 200), dtype=int)
    mask[:t_len, :t_len] = np.eye(t_len, dtype=int)
    # 设置后 t_con 行
    for i in range(c_len):
        row_index = t_len + i
        mask[row_index, :t_len + i + 1] = 1
    x = tokenizer.encode(combined_text_x, add_special_tokens=False, padding='max_length', truncation=True,
                         max_length=200)  # 将字转换成序号
    y = tokenizer.encode(combined_text_y, add_special_tokens=False, padding='max_length', truncation=True, max_length=200)
    # 将y的前t_len个元素转换为-100
    y = [-100] * t_len + y[t_len:]
    return x, y, mask


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, tokenizert, corpus):
    dataset_x = []
    dataset_y = []
    mask_all = []
    for i in range(sample_length):
        x, y, mask = build_sample(tokenizert, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        mask_all.append(mask)
    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)
    mask_all = np.array(mask_all)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), torch.LongTensor(mask_all)


# 建立模型
def build_model(char_dim, vocab, pretrain_model_path):
    model = LanguageModel(char_dim, vocab, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= window_size:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            mask = torch.LongTensor([0])
            y = model(mask, x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
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


# 计算文本ppl
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
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 50  # 训练轮数
    batch_size = 16  # 每次训练样本个数
    train_sample = 100  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    window_size = 100  # 样本文本长度
    vocab_size = 21128  # 字表大小

    pretrain_model_path = r"D:\AIpython学习\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_title_content(corpus_path)  # 加载语料
    model = build_model(char_dim, vocab_size, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, mask = build_dataset(batch_size, tokenizer, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(mask, x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("韩国7岁女童被性侵 总统向国民致歉", model, tokenizer, window_size))
        print(generate_sentence("邓亚萍：互联网要有社会担当", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    # croups = load_title_content("sample_data.json")
    train("sample_data.json", False)
