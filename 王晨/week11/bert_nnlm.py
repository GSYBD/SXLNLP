import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class BERT_NNLM(nn.Module):
    def __init__(self, bert_model_name):
        super(BERT_NNLM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state)

        if labels is not None:
            loss = self.criterion(logits.view(-1, self.bert.config.vocab_size), labels.view(-1))
            return loss, logits
        else:
            return logits


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.data[idx]['title']
        content = self.data[idx]['content']

        inputs = self.tokenizer(title, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt")
        targets = self.tokenizer(content, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt")

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()

        labels = targets.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_corpus(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def build_model(bert_model_name):
    return BERT_NNLM(bert_model_name)

def build_dataset(batch_size, tokenizer, max_len, corpus):
    dataset = CustomDataset(corpus, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def generate_sentence(input_text, model, tokenizer, max_length=50, temperature=1.0):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        generated = input_ids
        for _ in range(max_length):
            outputs = model(input_ids=generated, attention_mask=attention_mask)
            logits = outputs[:, -1, :] / temperature  # 应用温度调整
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)  # 使用softmax并采样
            generated = torch.cat((generated, next_token), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long)), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        generated_text = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
        return generated_text


def train(corpus_path, save_weight=True):
    epoch_num = 100
    batch_size = 128
    window_size = 50
    learning_rate = 0.001
    pretrain_model_path = r'E:\AI\课程资料\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    corpus = load_corpus(corpus_path)
    model = build_model(pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("开始训练")

    dataloader = build_dataset(batch_size, tokenizer, window_size, corpus)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in dataloader:
            x = batch['input_ids']
            y = batch['labels']
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss, _ = model(input_ids=x, attention_mask=None, labels=y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 生成测试
        print("生成文本示例:")
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer, max_length=50))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return
# Example usage
# train('/path/to/sample_data.json')
if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)
