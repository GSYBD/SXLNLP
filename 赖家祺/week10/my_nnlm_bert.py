# coding:utf8
import os
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

"""
使用bert结构完成自回归语言模型训练
"""
# class LanguageModel(nn.Module):
#     def __init__(self, input_size, vocab, pretrain_model_path):
#         ## input_size = char_dim = 256
#         ## len(vocab) = 3961
#         super().__init__()
#         self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=input_size) # 3961, 256
#         self.lstm = nn.LSTM(input_size,input_size, num_layers=1, batch_first=True)  # 256, 256
#         self.classify = nn.Linear(input_size, len(vocab))  ##  256,3961
#         self.dropout = nn.Dropout(p=0.2)
#         self.loss = nn.functional.cross_entropy
#
#     def forward(self, x, y=None):
#         x = self.embedding(x)
#         x, _ = self.lstm(x)
#         y_pred = self.classify(x)  # torch.Size([64, 10, 3961])
#
#         if y is not None:
#             return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1)) # 640, 3961
#         else:
#             return torch.softmax(y_pred, dim=-1) # 64, 10, 3961

class LanguageModelBert(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModelBert, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if y is not None:
            # 下三角mask矩阵
            x = x  # torch.Size([128, 10])
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)

def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# def build_model(vocab, char_dim):
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModelBert(char_dim, vocab, pretrain_model_path)
    return model

# def build_sample(vocab, window_size, corpus):
def build_sample(tokenizer, window_size, corpus):
    # 预训练好的Bert本身自带词表
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    # y = [vocab.get(word, vocab["<UNK>"]) for word in target]

    """
    - add_special_tokens=False: 这个参数指定是否在序列的开始和结束添加特殊的标记（如[CLS]和[SEP]），通常用于分类任务。这里设置为 False 表示不添加这些特殊标记。
    - padding='max_length': 这个参数指定了填充（padding）的策略。设置为 'max_length' 表示序列将被填充到 max_length 指定的长度，如果序列已经达到或超过这个长度，则会被截断。
    - truncation=True: 这个参数指定是否截断序列，如果序列长度超过了 max_length，它将被截断到这个长度。
    - max_length=window_size: 这是序列的最大长度。如果序列长度超过这个值，它将被截断；如果没有达到这个长度，序列将被填充到这个长度。
    """
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)

    return x, y

def build_dataset(sample_length, tokenizer, window_size, corpus):
    # sample_length = batch_size = 64
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x]) # torch.Size([1, 10])
            if torch.cuda.is_available():
                x = x.cuda()
            # y = model.forward(x) # torch.Size([1, 10, 3961])
            y = model.forward(x)[0][-1] # 3961
            index = sampling_strategy(y)
            # pred_char = reverse_vocab[index]
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
    epoch_num = 20
    batch_size = 128 # 64
    train_sample = 10000 # 500
    char_dim = 768   # 256
    window_size = 10
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率
    pretrain_model_path = r'D:\八斗人工智能\nlp\0-bert-base-chinese\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # vocab = build_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)
    # model = build_model(vocab, char_dim)
    model = build_model(vocab_size, char_dim, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print(f"第{epoch + 1}轮平均Loss: {np.mean(watch_loss):.4f}")
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("my_model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == '__main__':
    train("corpus.txt", save_weight=False)

