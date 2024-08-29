import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json

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
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            x, _ = self.bert(x,attention_mask=mask)
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
           title = line["title"]
           content = line["content"]
           corpus.append((title, content))
    return corpus


#随机生成一个样本
def build_sample(tokenizer,corpus,max_length):
    entry = random.choice(corpus)
    title, content = entry
    title_size = len(title)
    input_x = title + "[SEP]" + content
    output_y = content + "[EOS]"
    content_size = len(content)
    x = tokenizer.encode(input_x, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length+2)   #将字转换成序号
    y = tokenizer.encode(output_y, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length+2)
    y = [-100] *title_size + y[0:max_length+2-title_size]
    mask = generate_attention_mask_matrix(title_size,max_length+2)
    return x, y, mask


def generate_attention_mask_matrix(title_size, max_seq_length):
    # 初始化矩阵，所有元素都为0
    mask_matrix = [[0] * max_seq_length for _ in range(max_seq_length)]
            # 但实际上，我们可以更直接地生成这个矩阵
    for i in range(max_seq_length):
        if i <= title_size:
            for j in range(title_size+1):
                mask_matrix[i][j] = 1
        else:
            for j in range(i+1):
                mask_matrix[i][j] = 1
    return mask_matrix


#计算最大文本长度
def get_max_length(corpus):
    max_length = 0
    for entry in corpus:
        #print(entry)
        title, content = entry
        # 拼接title和content，并计算长度
        length = len(title) + len(content)
        # 更新最大长度（如果需要）
        if length > max_length:
            max_length = length
    return max_length


#建立数据集

def build_dataset(sample_length, tokenizer, max_length, corpus):
    dataset_x = []
    dataset_y = []
    dataset_mask = []
    for i in range(sample_length):
        x, y, mask = build_sample(tokenizer,corpus, max_length)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_mask.append(mask)
    # print(dataset_mask)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), torch.LongTensor(dataset_mask)

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
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



def train(corpus_path, save_weight=True):
    epoch_num = 9        #训练轮数
    batch_size = 10       #每次训练样本个数
    train_sample = 500   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率


    pretrain_model_path = r"D:\BaiduNetdiskDownload\八斗精品班\第六周 预训练模型\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)



    corpus = load_corpus(corpus_path)     #加载语料
    max_length = get_max_length(corpus)
    #print(max_length) 135 +1
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, mask = build_dataset(batch_size, tokenizer, max_length, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            # print(x.shape)
            # print(y.shape) 128 * 136
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # print(generate_sentence("北美洲发现肥皂人", model, tokenizer, window_size=10))
        # print(generate_sentence("下午茶•你，怕老吗？", model, tokenizer, window_size=10))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    train("sample_data.json", False)