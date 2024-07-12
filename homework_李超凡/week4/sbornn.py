import jieba
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


class TensorModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_rnn_layer, vocab):
        super(TensorModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, embedding_dim, padding_idx=0)  # len(vocab)+1的原因是padding不在词表中
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          batch_first=True,
                          num_layers=num_rnn_layer)
        self.classifition=nn.Linear(hidden_size,2)
        self.loss_func=nn.CrossEntropyLoss(ignore_index=-100) #计算交叉熵的时候忽略label的padding

    def forward(self,x,y=None):
        x=self.embedding(x) # (batch_size,max_length,input_dim)
        x,_=self.rnn(x) # (batch_size,max_length,hidden_size)
        y_pred=self.classifition(x) # (batch_size,max_length,2)
        if y is not None:
            return self.loss_func(y_pred.reshape(-1,2),y.view(-1))
        else:
            return y_pred


class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()

    def load(self):
        # 读取语料提取每一行文本
        self.data = []
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                # 文本举止转向量
                sequence = sentence_2_sequence(line,self.vocab)
                # 标注分词边界
                label = sentence_2_label(line)
                # 对sequence和label做padding
                sequence, label = padding(sequence, label,self.max_length)
                # 转为张量格式
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def sentence_2_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab["unk"]) for char in sentence]
    return sequence

def sentence_2_label(sentence):
    # 使用结巴分词获取分析序列
    word_ls = jieba.lcut(sentence)
    label = [0] * len(sentence)
    pos = 0
    for word in word_ls:
        pos += len(word)
        label[pos - 1] = 1
    return label

def padding(sequence, label, max_length):
    # 截断
    sequence = sequence[:max_length]
    label = label[:max_length]
    # padding
    sequence = [0] * (max_length - len(sequence)) + sequence
    label = [-100] * (max_length - len(label)) + label
    return sequence, label

def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index
    vocab["unk"] = len(vocab) + 1
    return vocab


def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab, max_length)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return data_loader


def train():
    epoch_num = 20
    batch_size=20
    embedding_dim=50
    hidden_size=100
    num_rnn_layer=1
    max_length=20
    learning_rate=1e-3
    vocab_path="chars.txt"
    corpus_path="corpus.txt"
    vocab=build_vocab(vocab_path)
    data_loader=build_dataset(corpus_path,vocab,max_length,batch_size)
    model=TensorModel(embedding_dim,hidden_size,num_rnn_layer,vocab)
    optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
    for epoch in tqdm(range(epoch_num),desc="epoch:"):
        model.train()
        watch_loss=[]
        for x,y in data_loader:
            optim.zero_grad() # 梯度归零
            loss=model.forward(x,y) # 计算损失
            loss.backward() # 反向传播
            optim.step() # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

    #保存模型
    torch.save(model.state_dict(), "model.pth")
    return

def predict(model_path,vocab_path,input_strings):
    embedding_dim=50
    hidden_size=100
    num_rnn_layers=1
    vocab=build_vocab(vocab_path)
    model=TensorModel(embedding_dim,hidden_size,num_rnn_layers,vocab)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for input_string in input_strings:
        x=sentence_2_sequence(input_string,vocab)
        with torch.no_grad():
            result=model.forward(torch.LongTensor([x]))[0]
            result=torch.argmax(result,dim=-1)
            # 在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()


if __name__ == "__main__":
    # train()
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]

    predict("model.pth", "chars.txt", input_strings)

