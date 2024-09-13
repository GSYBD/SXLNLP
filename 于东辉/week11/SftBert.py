import  torch
import  torch.nn as nn
import  numpy as np
import  math
import  random
import  os
import  re
from  transformers import BertTokenizer, BertModel
import json
from  torch.utils.data import Dataset, DataLoader

"基于Bert结构进行sft训练"
class BertSftModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(BertSftModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, mask=None, y=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return  torch.softmax(y_pred, dim=-1)
"将每句话的title和content写入corpus中"
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
        print("Corpus loaded with", len(corpus), "items.")  # 调试语句
    return corpus

def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        #注意x和y之间的错位构造，后面的answer进行了错位构造
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(prompt_encode)*[-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        "构建一个mask矩阵，目标是在prompt_encode之间可以进行交互，答案编码之间不能进行交互"
        mask = creat_mask(prompt_encode, answer_encode)
        #对x和y进行长度的确定截取或者填充
        x = x[:max_length] + [0]*(max_length-len(x))
        y = y[:max_length] + [0]*(max_length-len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)



def creat_mask(prompt, answer):
    len_p = len(prompt) + 2
    len_a = len(answer) + 1
    #进行张量掩码的创建
    mask = torch.ones(len_a + len_p, len_a +len_p)
    #设置p之间的掩码是可以相互进行关系获取的并且p的token不可以看到a的任何token
    for i in range(len_p):
        mask[i, len_p:] = 0
    #设置a之间的掩码只能看到之前的不能看到之后的
    for i in range(len_a):
        mask[len_p + i, len_p + i + 1:] = 0
    return mask

def pad_mask(tensor, target_shape):
    height, width = tensor.shape
    target_heigth, target_width = target_shape
    #创建一个全是0的张量
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    #计算那些区域需要填充
    h_end = min(height, target_heigth)
    w_end = min(width, target_width)
    result[:h_end, :w_end] = tensor[:h_end, :w_end]
    return  result

def build_model(char_dim, vocab, pretrain_model_path):
    model = BertSftModel(768, 21128, pretrain_model_path)
    return model

#文本生成代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sample_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def sample_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return  int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return  np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def main(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 32
    char_dim = 768
    max_length = 50
    vocab_size = 21128
    learning_size = 0.001

    pretrain_model_path = r'D:\BaiduNetdiskDownload\第六周 预训练模型\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)
    model = build_model(char_dim, vocab_size, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_size)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, mask, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=====\n第%d论平均权重loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("武汉中学生赴美旅游丢钱包 40天后被寄回", model, tokenizer))
        print(generate_sentence("传统与现代之间：一对藏族新人的结婚照", model, tokenizer))

    if not  save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    main("sample_data.json", False)