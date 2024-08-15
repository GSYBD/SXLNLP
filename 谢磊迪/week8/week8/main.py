from  create_datas import load_data
from  config import config
from  model import SiameseNetwork,choose_optimizer
import numpy as np
from  evaluate import Evaluator
import torch
def  main():
    loaddatas = load_data(config,type1='train',shuffle=True)
    model = SiameseNetwork(config)
    optim =choose_optimizer(config,model)
    loss_li=[]
    evaluator = Evaluator(config,model)
    for epoch in range(config['epoch_num']):
        model.train()
        for batch in loaddatas:
            optim.zero_grad()
            loss = model(sentence1 = batch[0],sentence2 =batch[1],sentence3 = batch[2])
            loss.backward()
            optim.step()
            loss_li.append(loss.item())
        print(f"第{epoch}轮的loss:{np.mean(loss_li)}")
        evaluator.eval(epoch)
        torch.save(model,f'./model_output/model{epoch}.pt')
if __name__ == '__main__':
    main()
