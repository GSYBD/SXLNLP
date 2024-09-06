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
from tqdm import tqdm
"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim):
        super(LanguageModel, self).__init__()
        self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.encoder = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        hidden_size = self.encoder.config.hidden_size
        vocab_size = self.encoder.config.vocab_size
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy
    
    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, attention_mask=None):
        if y is not None:
            x = self.encoder(x, attention_mask=attention_mask)[0]
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x = self.encoder(x)[0]
            y_pred = self.classify(x)  
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
    # 读取json
    lines = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            line['content'] += '\n'
            lines.append(line)
    return lines

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus, if_show=False):
    index = random.randint(0, len(corpus) - 1)  #随机选择一个文本
    question_length = len(corpus[index]['title'])
    window = corpus[index]['title'] + corpus[index]['content'][:window_size-question_length]  #前n-1个字作为输入
    target = corpus[index]['title'][1:] + corpus[index]['content'][:window_size-question_length+1]
        # 创建 attention_mask
    if if_show:
        print(corpus[index]['title'])
        print(corpus[index]['content'])
    question_mask = torch.ones((window_size, question_length))
    content_mask = torch.tril(torch.ones(window_size - question_length, window_size - question_length))
    padding_mask = torch.zeros((question_length, window_size - question_length))
    
    # 将 masks 拼接起来
    attention_mask = torch.cat((padding_mask, content_mask), dim=0)
    attention_mask = torch.cat((question_mask, attention_mask), dim=1)
    
    # print(window, target)
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)   #将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)
    y[:len(corpus[index]['title'])-1] = [-100] * (len(corpus[index]['title'])-1)
    return x, y, attention_mask

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(tokenizer, sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    attention_masks = []
    for i in range(sample_length):
        x, y, attention_mask= build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        attention_masks.append(attention_mask)
    attention_masks = torch.stack(attention_masks)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), attention_masks

#建立模型
def build_model(char_dim):
    model = LanguageModel(char_dim)
    return model

#文本生成测试代码
def generate_sentence(tokenizer, model, window_size, corpus):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    index = random.randint(0, len(corpus) - 1)  #随机选择一个文本
    question = corpus[index]['title']
    content = corpus[index]['content']
    print('title:', question)
    print('content:', content)
    pred_txt = ""
    with torch.no_grad():
        while len(pred_txt) <= 100:
            x_title = question+pred_txt
            x = tokenizer.encode(x_title, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
            pred_txt += pred_char
            if pred_char == '\n':
                break
    return pred_txt

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
def calc_perplexity(tokenizer, sentence, model, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = tokenizer.encode(window, add_special_tokens=False)
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = tokenizer.encode(target, add_special_tokens=False)
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 100        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    char_dim = 256        #每个字的维度
    window_size = 100      #样本文本长度
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(char_dim)    #建立模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in tqdm(range(int(train_sample / batch_size))):
            x, y, attention_masks = build_dataset(tokenizer, batch_size, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y, attention_masks = x.cuda(), y.cuda(), attention_masks.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, attention_masks)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence(tokenizer, model, window_size, corpus))
        print("=========\n")
        print(generate_sentence(tokenizer, model, window_size, corpus))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("../sample_data.json", False)
