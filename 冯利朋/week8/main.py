import torch.cuda
import os
import numpy as np
from loader import load_data
from model import SeeNetWork, choose_optimizer
from evaluate import Evaluate
from predict import Predict
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def main(config):
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    epoch_num = config['epoch']
    train_data = load_data(config['train_data_path'], config)
    model = SeeNetWork(config)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = choose_optimizer(config, model)
    evaluate = Evaluate(config, model, logger)
    for epoch in range(1, epoch_num + 1):
        model.train()
        watch_loss = []
        logger.info("开始训练第%d轮" % epoch)
        for index, batch_data in enumerate(train_data):
            if torch.cuda.is_available():
                batch_data = [c.cuda() for c in batch_data]
            s1, s2, target = batch_data
            loss = model(s1, s2, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        logger.info("agg loss=%f" % np.mean(watch_loss))
        acc = evaluate.eval(epoch)
    model_path = os.path.join(config['model_path'], f'epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_path)
    return model, acc

if __name__ == '__main__':
    from config import Config
    # main(Config)
    predict = Predict(Config, './model_output/epoch_10.pth')
    predict.pre("我想把我的亲情号给换一下")
