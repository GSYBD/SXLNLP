
import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, ModelHub, choose_optimizer, id_to_label
from evaluate import Evaluator
from loader import load_data


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

'''

Model Training Master Program
'''


'''
为什么设置seed？ 为了确保实验是可复现的
'''
seed = Config["sentence_config"]["seed"]
random.seed(seed)
np.random.seed(seed)
'''
PyTorch在CPU和GPU上都有独立的随机数生成器。
torch.manual_seed(seed)用于设置PyTorch在CPU环境下的随机数种子，
从而确保每次运行时产生的随机数（例如初始化神经网络的权重）是一致的。
'''
torch.manual_seed(seed)
'''
用于设置PyTorch在所有GPU上使用的随机数生成器的种子。
这样可以保证即使在使用多块GPU进行深度学习任务时，生成的随机数也是可重复的。
'''
torch.cuda.manual_seed_all(seed)


def main(config, model_name=None):
    # model save path
    if os.path.exists(config['model_path']) is False:
        os.mkdir(config['model_path'])
    
    
    train_data = load_data(config['train_data_path'], config)
    
    hub = ModelHub(model_name,Config)
    model = hub.model
    
    
    
    cuda_flag=torch.cuda.is_available()
    multi_gpu_flag = False
    if cuda_flag:
        logger.info("GPU is available, the model can be moved to GPU")
        device_ids = list(range(config['num_gpus']))
        if len(device_ids)>1:
            logger.info("Multi GPU distributed training is available")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            multi_gpu_flag=True
        model = model.cuda()
        
    
    
    optimizer = choose_optimizer(config, model)

    
    evaluator = Evaluator(config, model, logger)

    
    
    # training procedure
    
    for epoch in range(config['epoch']):
        model.train()
        logger.info("Epoch: {}".format(epoch))
        watch_loss = []
        for batch in train_data:
            if cuda_flag:
                batch = [d.cuda() for d in batch]
            input_ids, labels = batch
            
            optimizer.zero_grad()
            loss = model(input_ids, labels) # average cross entropy of each batch
            
            if multi_gpu_flag:
                loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
            
        print("epoch:{} loss: {}".format(epoch+1,np.mean(watch_loss)))
        evaluator.eval(epoch)
        
    
    # save model weights
    model_path = os.path.join(config['model_path'], 'epoch_%d.pth'% epoch)
    # torch.save(model.state_dict(), model_path)
    
    return model, train_data



def batch_train(config):
    with open("metrics.csv", "w", encoding='utf8') as f:
        f.close()
        
    model_names = ["bert", "lstm"]
    
    for model_name in model_names:
        main(config, model_name)



if __name__ == '__main__':
    from config import Config
    # model, train_data = main(Config, "bert")
    batch_train(Config)
    
    # input = [[12, 9, 8, 34, 5, 8, 98]]
    # input = torch.LongTensor(input)
    # input = input.cuda()
    # output = model(input)
    
    # for i in output[0]:
    #     print(id_to_label(i, Config), end = ', ')
    