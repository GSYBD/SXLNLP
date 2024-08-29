#coding:utf8

import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel

"""
基于pytorch的LSTM语言模型
"""

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


#加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            corpus.append(line)
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer,corpus, max_length=150):
    index = random.randint(0, len(corpus) - 1)
    title = corpus[index]["title"]
    content = corpus[index]["content"]

    input_seq = tokenizer.encode(title + "[SEP]" + content, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length)  # 输入序列
    output_seq = tokenizer.encode(("[PAD]" * len(title)) + content + "[CLS]", add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length)  # 输出序列
    mask11 = torch.ones(len(title), len(title))
    mask12 = torch.zeros(len(title), max_length - len(title))
    mask21 = torch.ones(max_length - len(title), len(title))
    mask22 = torch.tril(torch.ones(max_length - len(title), max_length - len(title)))
    masktop = torch.cat((mask11, mask12), dim=1)
    maskbottom = torch.cat((mask21, mask22), dim=1)
    mask = (torch.cat((masktop, maskbottom), dim=0))

    return input_seq, output_seq, mask.tolist()
# 建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, corpus):
    dataset_x = []
    dataset_y = []
    dataset_mask = []
    for i in range(sample_length):
        x, y, mask = build_sample(tokenizer, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_mask.append(mask)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), torch.FloatTensor(dataset_mask)

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    with torch.no_grad():
        openLength = len(openings)
        pred_char = ""
        #生成了换行符，或生成文本超过150字则终止迭代
        while pred_char != "\n" and len(openings) <= 150:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return openings[openLength:]

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
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 5000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    

    pretrain_model_path = r"E:\Learn\nlp\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, mask = build_dataset(batch_size, tokenizer, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print("居家办公:", generate_sentence("居家办公[SEP]", model, tokenizer))
        print("罗氏瞒报不良反应事件:", generate_sentence("罗氏瞒报不良反应事件[SEP]", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train('sample_data.json', False)
