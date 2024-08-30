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
import json


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
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1),ignore_index=0)
        else:
            return torch.softmax(y_pred, dim=-1)

#加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            corpus.append({'x':sample['title'],'y':sample['content']})
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(corpus,config):
    max_length = config["max_length"]
    sample = random.choice(corpus)
    x,y = sample['x'],sample['y']
    x = tokenizer.encode(x,add_special_tokens=False)
    y = tokenizer.encode(y,add_special_tokens=False)
    ask_len = len(x)
    # xy拼接
    x += y
    sentence_len = len(x)

    # seq2seq mask
    mask = np.zeros((max_length,max_length),dtype=int)
    for i in range(sentence_len if sentence_len < max_length else max_length):
        if (i < ask_len):
            mask[:sentence_len,i] = 1
        else:
            mask[i:sentence_len,i] = 1
    
    # 手动padding
    if sentence_len < max_length:
        x = x + [0] * int(max_length - sentence_len)
    else:
        x = x[:max_length]

    # # 直接用max_length，然后前面追加len(ask)个0，然后截[:128]。
    #     # 如果实际内容不足128 ： [0,0,0,0,0,context_id,0,0]
    #     # 如果超长： [0,0,0,0,0,sub_context_id]
    y = np.copy(x[1:])

    # 给问题加padding,用于后续算loss
    # 需要错一个位
    y[:(ask_len -1)] = 0
    y = np.append(y,0)
    return x,y[:max_length].tolist(),mask 

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#corpus 语料字符串
def build_dataset(sample_length,corpus,config):
    dataset_x = []
    dataset_y = []
    dataset_mask = []
    for i in range(sample_length):
        x, y,mask = build_sample(corpus,config)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_mask.append(mask)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y),torch.IntTensor(dataset_mask)

#建立模型
def build_model(config):
    model = LanguageModel(config)
    return model

#文本生成测试代码
def generate_sentence(openings, model, config):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过100字则终止迭代
        while len(openings) <= 100:
            # Q:为什么是加上一轮结果再预测，而不是这一轮预测后直接添加，再往下轮判断是否继续。
            # A:因为训练数据和样本都是max_length。如果当前预测后直接相加，y_pred长度就是max_length+1了。
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

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])   #建立优化器
    print("文本词表模型加载完毕，开始训练")

    # 构建mask
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y,mask = build_dataset(batch_size, corpus,config) #构建一组训练样本
            if torch.cuda.is_available():
                x, y,mask = x.cuda(), y.cuda(),mask.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y,mask = mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("用别人的卡取钱 是提醒还是偷盗？", model, config))
        print(generate_sentence("标普上调三家中国银行 下调全球多家银行评级", model, config))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(Config,save_weight=False)
