import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TFBertModel, BertTokenizer
import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np

# 读取 CSV 文件
data = pd.read_csv('文本分类练习.csv')
data.columns = ['label', 'review']

# 加载 BERT 的分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 对数据进行分词
def tokenize_texts(texts, max_len):
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
    return encodings

max_len = 50
encodings = tokenize_texts(data['review'].tolist(), max_len)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(encodings['input_ids'], data['label'], test_size=0.2, random_state=42)
# print(y_test)
class TorchModel(nn.Module):
    def __init__(self, input_dim):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(input_dim, 2)
        self.activation = torch.sigmoid     #sigmoid做激活函数
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        sequence_output, pooler_output = self.bert(x)
        x = self.classify(pooler_output)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(torch.FloatTensor(y_pred), torch.LongTensor(y.squeeze()))
        else:
            return torch.FloatTensor(y_pred)

def evaluate(model, x_test, y_test):
    model.eval()
    print(f'正样本数量{sum(y_test)},负样本数量{len(y_test)}')
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x_test)      #模型预测
        print(y_pred)
        for y_p, y_t in zip(y_pred, y_test):  # 与真实标签进行对比
            if torch.argmax(y_p) >= 0.5 and y_t == 1:
                correct += 1
            elif torch.argmax(y_p) < 0.5 and y_t == 0:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main(x_train, y_train):
    epoch_num = 5        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = len(x_train)   #每轮训练总共训练的样本总数
    char_dim = 768         #每个字的维度
    model = TorchModel(char_dim)    #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)   #建立优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size
            x_train = x_train[batch_start:batch_end]
            y_train = y_train[batch_start:batch_end]
            optim.zero_grad()    #梯度归零
            loss = model(x_test, y_test)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, x_test, y_test)
        log.append([acc, np.mean(watch_loss)])
        torch.save(model.state_dict(), "model.pth")


if __name__ == '__main__':
    main(x_test, y_test)
