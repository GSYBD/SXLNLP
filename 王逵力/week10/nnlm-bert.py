import torch
from transformers import BertTokenizer, BertForMaskedLM
import os
import random

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, 'r', encoding='gbk') as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = tokenizer(window, return_tensors="pt", truncation=True, padding=True, max_length=window_size).to(device)
    y = tokenizer(target, return_tensors="pt", truncation=True, padding=True, max_length=window_size).to(device)
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return dataset_x, dataset_y

# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, device, window_size, max_length=30):
    model.eval()
    with torch.no_grad():
        pred_text = ""
        current_text = openings
        while len(current_text) < max_length:
            inputs = tokenizer(current_text, return_tensors="pt", truncation=True, padding=True, max_length=window_size).to(device)
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            next_token = tokenizer.decode([next_token_id])
            current_text += next_token
            pred_text = current_text
    return pred_text

# 计算文本ppl
def calc_perplexity(sentence, model, tokenizer, device):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=len(sentence)).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

def train(corpus_path, model, save_weight=True):
    epoch_num = 20        # 训练轮数
    batch_size = 64       # 每次训练样本个数
    train_sample = 50000   # 每轮训练总共训练的样本总数
    window_size = 10       # 样本文本长度
    vocab = {}             # BERT使用tokenizer，不需要构建词汇表
    corpus = load_corpus(corpus_path)     # 加载语料

    optim = torch.optim.Adam(model.parameters(), lr=0.01)   # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus) # 构建一组训练样本
            optim.zero_grad()    # 梯度归零
            for i in range(len(x)):
                outputs = model(**x[i], labels=y[i]['input_ids'])  # 计算loss
                loss = outputs.loss
                loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, device, window_size))
        print(generate_sentence("李慕站在山路上，深深地呼吸", model, tokenizer, device, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    # 确保你的corpus.txt文件存在并且路径正确
    train("corpus.txt", model, False)