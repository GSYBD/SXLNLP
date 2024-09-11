import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertModel



bert_path = r"D:\NLP\WEEK11\bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_path)


class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)

        self.classify = nn.Linear(self.bert.config.hidden_size, len(vocab))
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):

        x, _ = self.bert(x, attention_mask=mask)  # output shape:(batch_size, sen_len, input_dim)

        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)

        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab():
    vocab = tokenizer.get_vocab()
    return vocab


class DataGenerator:
    def __init__(self, data_path, max_length, vocab):
        self.config = {}
        self.path = data_path
        self.vocab = vocab
        self.config["max_length"] = max_length
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]

                self.prepare_data(title, content)
        return

    # 文本到对应的index
    # 头尾分别加入[cls]和[sep]
    def encode_sentence(self, text, max_length, with_cls_token=True, with_padding=False):
        input_id = []
        if with_cls_token:
            input_id.append(self.vocab["[CLS]"])

        if with_padding:
            input_id.extend(tokenizer.encode(text, max_length=max_length, padding='max_length',
                                             truncation=True, add_special_tokens=False))
        else:
            input_id.extend(tokenizer.encode(text, truncation=True, add_special_tokens=False))

        return input_id

    # 输入输出转化成序列
    def prepare_data(self, title, content):

        input_seq = self.encode_sentence(title, self.config["max_length"], False)  # 输入序列

        output_seq = self.encode_sentence(content, self.config["max_length"] - len(input_seq) - 1, True,
                                          True)  # 输出序列

        gold = self.encode_sentence(content + '[SEP]', self.config["max_length"] - len(input_seq),
                                    False,
                                    True)  # 不进入模型，用于计算loss

        # 拼接输入输出
        input_id = input_seq + output_seq
        output_id = [-1 for _ in range(len(input_seq))] + [-1 if x == 0 else x for x in gold]

        # 写mask
        s11 = torch.ones([len(input_seq), len(input_seq)])
        s12 = torch.zeros([len(input_seq), len(output_seq)])
        s21 = torch.ones([len(output_seq), len(input_seq)])
        s22 = torch.tril(torch.ones([len(output_seq), len(output_seq)]))

        mask = torch.zeros(self.config["max_length"], self.config["max_length"])
        mask[:len(input_seq), :len(input_seq)] = s11
        mask[:len(input_seq), len(input_seq):self.config["max_length"]] = s12
        mask[len(input_seq):self.config["max_length"], :len(input_seq)] = s21
        mask[len(input_seq):self.config["max_length"], len(input_seq):self.config["max_length"]] = s22

        self.data.append([torch.LongTensor(input_id),
                          torch.LongTensor(output_id),
                          mask])

        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 建立模型
def build_model(vocab):
    model = LanguageModel(vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了[SEP]，或生成文本超过70字则终止迭代
        while pred_char != "[SEP]" and len(openings) <= 70:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)

            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()

            y = model(x)[0][-1]

            index = sampling_strategy(y)
            # pred_char = reverse_vocab[index]
            pred_char = ''.join(tokenizer.decode(index))
            print(pred_char)
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


def train(corpus_path, save_weight=True):
    epoch_num = 40  # 训练轮数
    batch_size = 32  # 每次训练样本个数

    window_size = 200  # 样本文本长度
    vocab = build_vocab()  # 建立字表

    dg = DataGenerator(corpus_path, window_size, vocab)
    train_data = DataLoader(dg, batch_size=batch_size, shuffle=True)

    model = build_model(vocab)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            x, y, mask = batch_data  # 构建一组训练样本

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y, mask)  # 计算loss
            print('loss', loss)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        result = generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换", model)
        print(result, len(result))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)
