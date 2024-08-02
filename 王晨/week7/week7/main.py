import torch
import numpy as np
from config import Config
from model import TorchModel
from evaluate import evaluate
from loader import load_data
import os

def main():
    x_train, x_test, y_train, y_test = load_data('文本分类练习.csv', Config["max_len"], Config["test_size"], Config["seed"])
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = TorchModel(Config["char_dim"])
    optim = torch.optim.Adam(model.parameters(), lr=Config["lr"])
    log = []
    
    for epoch in range(Config["epoch_num"]):
        model.train()
        watch_loss = []
        train_sample = len(x_train)
        for batch in range(int(train_sample / Config["batch_size"])):
            batch_start = batch * Config["batch_size"]
            batch_end = batch_start + Config["batch_size"]
            x_batch = x_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            optim.zero_grad()
            loss = model(x_batch, y_batch)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        
        print("=========第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, x_test, y_test)
        log.append([acc, np.mean(watch_loss)])
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

if __name__ == '__main__':
    main()
