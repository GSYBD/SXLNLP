import os.path

from loader import load_data
from model import TorchModel, choose_optimizer
from evaluate import Evaluate
import torch
import numpy as np
import logging
from config import Config
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def main(config):
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    epoch_num = config['epoch']
    train_data = load_data(config['train_data_path'], config)
    model = TorchModel(config)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = choose_optimizer(config, model)
    evaluate = Evaluate(config, model, logger)
    for epoch in range(1, epoch_num + 1):
        logger.info("开始训练第%d轮" % epoch)
        watch_loss = []
        model.train()
        for index, batch_data in enumerate(train_data):
            if torch.cuda.is_available():
                batch_data = [c.cuda() for c in batch_data]
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        logger.info("agv loss = %f" % (np.mean(watch_loss)))
        acc = evaluate.eval(epoch)
    model_path = os.path.join(config['model_path'], f'epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == '__main__':

    main(Config)