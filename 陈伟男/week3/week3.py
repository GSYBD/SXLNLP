import torch
import torch.nn as nn
import numpy as np
import random
import string
def generate_text_sample():
    # 生成一个随机长度的字符串（3到10个字符）
    length = random.randint(3, 10)
    text = ''.join(random.choices(string.ascii_lowercase, k=length))
    return text

def classify_text(text):
    # 根据文本中是否包含字母"a"和/或"b"来分配类别
    has_a = 'a' in text
    has_b = 'b' in text
    if has_a and has_b:
        return 2
    elif has_a:
        return 0
    elif has_b:
        return 1
    else:
        return 3

def generate_dataset(size):
    dataset = []
    while len(dataset) < size:
        text = generate_text_sample()
        category = classify_text(text)
        dataset.append((text, category))
    return dataset

# 生成训练集和验证集
train_set = generate_dataset(600)
validation_set = generate_dataset(600)

charts = 'abcdefghijklmnopqrstuvwxyz'

def create_charts():
    map = {
        'pad': 0,
        'unk': len(charts)
    }
    for index, item in enumerate(charts):
        map[item] = index + 1
    return map


class TorchModel(nn.Module):
    def __init__(self, charts, data_size, hidden_size,):
        super(TorchModel, self).__init__()
        self.embadding = nn.Embedding(len(charts), data_size, padding_idx=0)
        self.rnn = nn.RNN(data_size, data_size, bias=False, batch_first=True)
        self.layer = nn.Linear(data_size, hidden_size)
        self.loss = nn.CrossEntropyLoss()


    def forward(self, x, y=None):
        # x是 20 * 10
        input = self.embadding(x) # 20 * 10 * 5
        _, rnn_output = self.rnn(input) # 20 * 10 * 5
        rnn_output = rnn_output.squeeze(0) # 20 * 5
        rnn_output = self.layer(rnn_output) # 20 * 4
        if y is not None:
            return self.loss(rnn_output, y)
        else:
            return rnn_output
            

def strToNumber(list, charts):
    result = []
    for str in list:
        item = [charts[x] for x in str]
        if len(item) < 10:
            item = item + [0] * (10 - len(item))
        else:
            item = item[0 : 11]
        result.append(item)
    return result


def evaluate(model, charts):
    model.eval()
    str_list = [x[0] for x in validation_set]
    input_x = torch.LongTensor(strToNumber(str_list, charts))
    input_y = torch.LongTensor([y[1] for y in validation_set]) 
    output_y = model(input_x)
    max_indices = torch.argmax(output_y, dim=-1)
    equal_elements = torch.eq(max_indices, input_y)
    num_equal_elements = torch.sum(equal_elements).item()
    acc = num_equal_elements / len(validation_set)
    print("acc:%.2f" % acc)


def main():
    hidden_size = 4
    data_size = 20
    batch_size = 20
    train_sample = 600
    epoch_num = 300
    learning_rate = 0.001
    charts = create_charts()
    model = TorchModel(charts, data_size, hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epoch_num):
        watch_loss = []
        model.train()
        for batch in range(int(train_sample / batch_size)):
            batch_list = train_set[batch * batch_size : (batch + 1) * batch_size]
            str_list = [x[0] for x in batch_list]
            input_x = torch.LongTensor(strToNumber(str_list, charts))
            input_y = torch.LongTensor([y[1] for y in batch_list])
            optim.zero_grad()
            loss = model(input_x, input_y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        evaluate(model, charts)


           

if __name__ == "__main__":
    main()