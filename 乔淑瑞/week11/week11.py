import json
import os
from random import random

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class LanguageModel(nn.Module):
    def __init__(self, pretrain_model_path, hidden_size, vocab_size):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y, mask):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


def load_corpus(corpus_path):
    corpus = []
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])

    return corpus


def create_mask(prompt_len, answer_len):
    len1 = prompt_len + 2
    len2 = answer_len + 1
    mask = torch.ones(len1 + len2, len1 + len2)
    for i in range(len1):
        mask[i, len1:] = 0
    for i in range(len2):
        mask[len2 + i, len2 + i + 1:] = 0
    return mask


def pad_mask(mask, target_shape):
    height, width = mask.shape
    target_height, target_width = target_shape
    result = torch.zeros(target_shape, dtype=mask.dtype, device=mask.device)
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    result[:h_end, :w_end] = mask[:h_end, :w_end]
    return result


def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)

        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [
            tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]

        # 构建mask
        mask = create_mask(len(prompt_encode), len(answer_encode))

        # padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)

        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])

    return dataset

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


def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #生成文本超过30字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def main(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    char_dim = 768  # 每个字的维度
    max_length = 50  # 样本文本长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r'D:\material\八斗\第六周 预训练模型\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    corpus = load_corpus(corpus_path)  # 加载语料
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)  # 建立数据集
    model = LanguageModel(pretrain_model_path, char_dim, vocab_size)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), learning_rate)  # 建立优化器
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data: #构建一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return