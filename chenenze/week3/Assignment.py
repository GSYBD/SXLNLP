import torch
import torch.nn as nn
import numpy as np
import random
import json

voc_dict = json.load(open('./dict.json', 'r'))
vocab = "abcdefgh12"

class Classification_Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Classification_Model, self).__init__()
        self.embedding = nn.Embedding(len(voc_dict), input_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.activation = nn.functional.softmax
        self.output = nn.Linear(hidden_size, input_size)
        self.loss = nn.functional.cross_entropy
    
    def forward(self, x, y=None):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.output(x)
        y_pred = self.activation(x, dim=1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    length = 10
    # choose a random letter among 26 letters
    x = [random.choice(vocab) for _ in range(length)]
    X = [voc_dict[c] if c in voc_dict else voc_dict['unk'] for c in x]
    # times of 'a' in x
    y = [v == x.count('a') for v in range(len(x))]
    return X, y, x

def build_dataset(total_sample_num, device):
    X = []
    Y = []
    vocs = []
    for i in range(total_sample_num):
        x, y, voc = build_sample()
        X.append(x)
        Y.append(y)
        vocs.append(voc)
    return torch.LongTensor(X).to(device), torch.FloatTensor(Y).to(device), vocs

def evaluate(model, device):
    model.eval()
    N = 1000
    X, Y, _ = build_dataset(N, device)
    with torch.no_grad():
        Y_pred = model(X)
        acc = (Y_pred.argmax(dim=1) == Y.argmax(dim=1)).float().mean().item()
        print(f"Accuracy: {acc}")
    return acc

def predict(model_path, x, device):
    model = Classification_Model(10, 20)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    X = []
    y = []
    for d in x:
        X.append([voc_dict[c] if c in voc_dict else voc_dict['unk'] for c in d] + [3] * (10 - len(d)))
        y.append(d.count('a'))
    with torch.no_grad():
        y_pred = model.forward(torch.LongTensor(X).to(device))
    for i in range(len(x)):
        print(f"Input: {x[i]}, Prediction: {y_pred[i].argmax().item()}, Probability: {y_pred[i].max().item()},Ground Truth: {y[i]}")
    return y_pred

def main(device):
    model = Classification_Model(10, 20)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epoch = 1000
    samples = 5000
    bach_size = 100
    
    for i in range(num_epoch):
        model.train()
        X, Y, _ = build_dataset(samples, device)
        train_loss = 0
        for j in range(0, samples, bach_size):
            X_batch = X[j:j + bach_size]
            Y_batch = Y[j:j + bach_size]
            loss = model(X_batch, Y_batch)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * bach_size / samples
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {train_loss}")
            acc = evaluate(model, device)
        if acc > 0.99: 
            break

    torch.save(model.state_dict(), "model.pth")
    return

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # main(device)
    
    test_data = ["abaaaa", "abc456", "cdef", "afabc"]
    predict("model.pth", test_data, device)