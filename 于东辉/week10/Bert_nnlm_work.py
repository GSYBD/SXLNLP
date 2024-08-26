import  torch
import  torch.nn as nn
import  numpy as np
import  math
import  random
import  os
import  re
from  transformers import BertTokenizer, BertModel

"使用Bert进行词生成，要求使用mask掩码对Bert的遮盖，使其只能从前获取语句的关系数值"
"建立模型类，具体的输入参数以及模型前向运算函数"
class BertMaskModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(BertMaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)


    def forward(self, x, y=None):
        if y is not None:
            #构建训练时使用的下三角掩码
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
               mask = mask.cuda()
            x,_ = self.bert(x, attention_mask=mask) #batch_size,seq_len,vocab_size
            y_pred = self.classify(x) #batch_size, vocab_size
            return nn.functional.cross_entropy(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #这里我们不输入y，所以我们得到的是预测值的概率分布
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


#加载语料--开始采药
"将这些预料进行切割，得到一个都是符号的变量"
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return  corpus

#随机生成一个样本---将药材切成合适的大小并研磨成药粉
#要求前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus)-window_size-1)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start+1:end+1]

    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

    return  x, y


#建立数据集---将药材打包
#将之前建立的随机样本进行分词然后将其转换成tensor张量
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dateset_x = []
    dateset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dateset_x.append(x)
        dateset_y.append(y)
    return  torch.LongTensor(dateset_x), torch.LongTensor(dateset_y)

#建立模型--丹炉构建
def build_model(vocab, char_dim, pretrain_model_path):
    model = BertMaskModel(768, 21128, pretrain_model_path)
    return model

#文本生成代码--练习制作丹药
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #设置迭代标准，生成换行符或者生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <=30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return  openings


#进行预测策略选择--选择成丹手法
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return  int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)





#对数据进行一个训练
def train(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 128
    train_sample = 10000
    char_dim = 768
    vocab_size = 21128
    window_size = 10
    learning_rate = 0.001

    pretrain_model_path = r'D:\BaiduNetdiskDownload\第六周 预训练模型\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)
    model = build_model(vocab_size, char_dim, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("文本此表模型加载完毕，开始训练")
    for epoch in  range(epoch_num):
        model.train()
        watch_loss=[]
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("====\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("矮个子捕快看了看前方凸起的", model, tokenizer, window_size))
        print(generate_sentence("高个子吞了口唾沫，说道：", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "path")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return
if __name__ == "__main__":
    train("corpus.txt", False)