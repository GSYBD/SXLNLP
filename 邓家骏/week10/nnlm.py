#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel
from config import Config
from transformers import BertTokenizer


"""
基于pytorch的LSTM语言模型
"""

tokenizer = BertTokenizer.from_pretrained(Config["bert_path"])

class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(config['bert_path'],num_hidden_layers=config['num_layers'],return_dict=False)
        self.classify = nn.Linear(config['hidden_size'], config['vocab_size'])
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None ,mask=None):
        x,_ = self.bert(x,attention_mask= mask)
        y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def process_sentence(sentence,config):
    input_ids = tokenizer.encode(sentence, max_length=config["max_length"] + 1, truncation=True, padding='max_length', add_special_tokens=False)
    # new_input = input_ids[1:-1]
    special_marks = tokenizer.encode('。！？',add_special_tokens=False)
    x = []
    y = []
    for id in input_ids:
        x.append(id)
        y.append(id)
        if id in special_marks:
            x.append(tokenizer.cls_token_id)
            y.append(tokenizer.sep_token_id)
    return x[:30], y[1:31]

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(window_size, corpus,config):
    start = random.randint(0, len(corpus) - 2 - window_size)
    
    end = start + window_size + 1
    window = corpus[start:end] #同时取x,y
    x,y = process_sentence(window,config)
    return x,y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, window_size, corpus,config):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus,config)
        dataset_x.append(x)
        dataset_y.append(y)
    dataset_x
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(config):
    model = LanguageModel(config)
    return model

# 
def generate_left2right_mask(batch_size,seq_length):
    mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.int)).unsqueeze(0).expand(batch_size, -1, -1)
    return mask



#文本生成测试代码
def generate_sentence(openings, model, config):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while len(openings) <= 30:
            # 为什么是加上一轮结果再预测，而不是这一轮预测后直接添加，再往下轮判断是否继续。
            # 因为训练数据和样本都是max_length。如果当前预测后直接相加，y_pred长度就是max_length+1了。
            openings += pred_char
            encoded_inputs = tokenizer(openings, padding='max_length', max_length=config["max_length"] , truncation=True, return_tensors="pt",add_special_tokens=False)
            x = encoded_inputs['input_ids']
            attention_mask = encoded_inputs['attention_mask']

            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()
            y = model(x,mask = attention_mask)[0][-1]
            index = sampling_strategy(y)
            if index == tokenizer.sep_token_id:
                break
            # if index in [tokenizer.pad_token_id,tokenizer.unk_token_id,tokenizer.mask_token_id]:
            #     continue
            pred_char = tokenizer.decode(index,skip_special_tokens=True)

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


def train(config,save_weight=True):
    corpus_path = config['corpus_path']
    epoch_num =  config['epoch']       #训练轮数
    batch_size = config['batch_size']        #每次训练样本个数
    train_sample = config['train_sample']   #每轮训练总共训练的样本总数
    char_dim = config['hidden_size']        #每个字的维度
    window_size = config['max_length']       #样本文本长度
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(config)    #建立模型
    mask = generate_left2right_mask(batch_size,config['max_length'])

    if torch.cuda.is_available():
        model = model.cuda()
        mask = mask.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])   #建立优化器
    print("文本词表模型加载完毕，开始训练")

    # 构建mask
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, window_size, corpus,config) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零



            loss = model(x, y,mask = mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, config))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, config))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        print(base_name)
        model_path = os.path.join(r"D:\code\data\week10_data", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(Config)
