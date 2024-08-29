import json
import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"E:\BaDou\第6周 预训练模型\bert-base-chinese", return_dict=False)
        self.dropout = nn.Dropout(0.1)
        self.classify = nn.Linear(input_dim, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None, mask=None):
        x, _ = self.bert(x, attention_mask=mask)
        x = self.dropout(x)
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

def load_corpus(path):
    titles, contents = [], []
    with open(path, encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            titles.append(data["title"])
            contents.append(data["content"])
    return [titles, contents]

def build_dataset(sample_length, tokenizer, corpus, max_len):
    dataset = [build_sample(tokenizer, corpus, max_len) for _ in range(sample_length)]
    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

def create_mask(question_size, answer_size):
    len_s1 = question_size + 2
    len_s2 = answer_size + 1
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    mask[:len_s1, len_s1:] = 0
    mask[len_s1:, len_s1:] = torch.triu(torch.ones(len_s2, len_s2), diagonal=1)
    return mask

def pad_mask(tensor, target_shape):
    height, width = tensor.shape
    target_height, target_width = target_shape
    # 创建一个全零张量,形状为目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 计算需要填充或截断的区域
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    result[:h_end, :w_end] = tensor[:h_end, :w_end]
    return result


def build_sample(tokenizer, corpus, max_len, valid_flag=False):
    x_list, y_list = corpus
    random_index = random.randint(0, len(x_list) - 1)
    x = x_list[random_index]
    if valid_flag:
        print(x)
    y = y_list[random_index]
    input_ids_x = tokenizer.encode(x, add_special_tokens=False)
    input_ids_y = tokenizer.encode(y, add_special_tokens=False)
    pad_x = [tokenizer.cls_token_id] + input_ids_x + [tokenizer.sep_token_id] + input_ids_y

    pad_y = len(input_ids_x) * [-1] + [-1] + input_ids_y
    pad_x = pad_x[:max_len] + [0] * (max_len - len(pad_x))
    pad_y = pad_y[:max_len] + [0] * (max_len - len(pad_y))

    mask = create_mask(len(input_ids_x), len(input_ids_y))
    mask = pad_mask(mask, (max_len, max_len))
    return [torch.LongTensor(pad_x), torch.LongTensor(pad_y), mask]

def build_model(vocab_size, char_dim):
    return LanguageModel(char_dim, vocab_size)

def sampling_strategy(prob_distribution):
    strategy = "greedy" if random.random() > 0.1 else "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(len(prob_distribution), p=prob_distribution)

def evaluate(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y_pred = model(x)[0][-1]
            index = sampling_strategy(y_pred)
            openings.append(index)
    return tokenizer.decode(openings)

def train(corpus_path, save_weight=True):
    epoch_num = 15
    batch_size = 32
    train_sample = 1000
    char_dim = 768
    tokenizer = BertTokenizer.from_pretrained(r"E:\BaDou\第6周 预训练模型\bert-base-chinese")
    vocab_size = 21128
    max_len = 50
    corpus = load_corpus(corpus_path)
    model = build_model(vocab_size, char_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = build_dataset(train_sample, tokenizer, corpus, max_len)
    print("模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y, mask in dataset:
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()
            loss = model(x, y, mask)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print(f"第{epoch + 1}轮平均loss: {avg_loss:.6f}")
        result1 = evaluate("姑娘钱包跌落西湖 小伙冒寒入水捞回", model, tokenizer)
        print(result1)

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train("sample_data.json", save_weight=False)
