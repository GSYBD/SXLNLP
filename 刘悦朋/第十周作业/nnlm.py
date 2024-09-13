import torch
import torch.nn as nn
import numpy as np
import random
import os
import re
from transformers import BertModel, BertTokenizerFast

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.vocab = BertTokenizerFast.from_pretrained('../bert-base-chinese')  # 字表
        self.bert = BertModel.from_pretrained('../bert-base-chinese', return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)  # 768 21128
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x, _ = self.bert(x)  # output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    corpus = re.sub('[^\u4e00-\u9fa5]', '', corpus)
    return corpus


def build_sample(model, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = model.vocab(window).input_ids
    y = model.vocab(target).input_ids
    if len(x[1: -1]) != 10:
        print(x[1: -1])
    if len(y[1: -1]) != 10:
        print(y[1: -1])
    return x[1: -1], y[1: -1]


def build_dataset(sample_length, model, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(model, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 文本生成测试代码
def generate_sentence(openings, model):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while len(openings) <= 30:
            openings += pred_char
            x = model.vocab(openings).input_ids
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = model.vocab.decode(index)
    return openings


def sampling_strategy(prob_distribution):
    if random.random() > 0:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train(corpus_path, save_weight=False):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    window_size = 10  # 样本文本长度
    corpus = load_corpus(corpus_path)  # 加载语料
    model = LanguageModel()  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, model, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, sum(watch_loss) / len(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join(base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("corpus.txt")
    # language_model = LanguageModel()
    # if torch.cuda.is_available():
    #     language_model = language_model.cuda()
    # corpus = load_corpus('corpus.txt')  # 2006975 -> 1726268
    # print(len(corpus))

    # [4937, 8024, 2303, 3209, 2347, 5307, 6672, 1057, 6863, 1265]
    #       [8024, 2303, 3209, 2347, 5307, 6672, 1057, 6863, 1265, 1914]
    # x, y = build_sample(model, 10, corpus)

    # print(model.vocab.decode([5710, 7676,  511, 5018, 9894, 4995, 2514]))
    # print(model.vocab.decode([511, 679, 1962, 8024, 704, 6369, 749, 8013, 8121]))
    # print(model.vocab.decode([679, 1962, 8024, 704, 6369, 749, 8013, 8522]))

    # print(generate_sentence('李慕站在山路上，深深的呼吸', language_model))
