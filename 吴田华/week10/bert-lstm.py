import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

class BertLanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(BertLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.position_embedding = nn.Embedding(512, input_dim)  # Max length of 512
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8), num_layers=6)
        self.classify = nn.Linear(input_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        x = self.transformer(x)
        y_pred = self.classify(x)
        return y_pred

    def compute_loss(self, y_pred, y):
        return F.cross_entropy(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))

# The rest of the functions remain largely the same
# Just replace the model creation and training functions

def build_model(vocab, char_dim):
    model = BertLanguageModel(char_dim, len(vocab))
    return model

def train(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 64
    train_sample = 50000
    char_dim = 256
    window_size = 10
    vocab = build_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)
    model = build_model(vocab, char_dim)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    print("文本词表模型加载完毕，开始训练")
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            y_pred = model(x)
            loss = model.compute_loss(y_pred, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    
    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train("corpus.txt", False)