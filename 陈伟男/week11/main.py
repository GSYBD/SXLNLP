from config import Config
import os
from loader import get_loader
from model import Model
from torch.optim import Adam
from tqdm import tqdm
import torch

def main():
  if not os.path.isdir(Config["model_path"]):
        os.mkdir(Config["model_path"])
  train_loader, train_generator = get_loader(Config['train_path'])
  model = Model()
  optimizer = Adam(model.parameters(), lr=Config['lr'])
  for epoch in range(Config['epochs']):
      train_bar = tqdm(train_loader)
      model.train()
      loss = 0
      for index, data in enumerate(train_bar):
          inputs, labels = data
          optimizer.zero_grad()
          loss = model(inputs, labels, train_generator.attention_mask)
          loss.backward()
          optimizer.step()
          train_bar.set_description("Epoch: {}/{} | Loss: {:.4f}".format(epoch+1, Config['epochs'], loss.item()))
          loss += loss.item()
      print("Epoch: {}/{} | Loss: {:.4f}".format(epoch+1, Config['epochs'], loss/len(train_loader)))
      if (epoch+1) % Config['save_interval'] == 0:
          torch.save(model.state_dict(), os.path.join(Config["model_path"], "model_{}.pth".format(epoch+1)))
     
if __name__ == "__main__":
  main()