import os
import random

import numpy as np
import torch
from torch.optim import Adam, SGD
import logging

from transformers import BertTokenizer

from model import SftModel
from loader import load_data

####################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


####################################

def mian(config):
    ####################################
    #(1) 创建模型参数路径
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    ####################################
    #(2) 创建模型对象
    model = SftModel(config)
    ####################################
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    ####################################
    #(3) 加载优化器
    optim = choose_optimizer(config, model)

    #(4) 加载训练数据
    train_data = load_data(config['train_data_path'], config)

    #(5) 开始训练
    for epoch in range(config['epochs']):
        epoch += 1
        ####################################
        model.train()
        if cuda_flag:
            model.cuda()
        logger.info("epoch %d begin" % epoch)
        ####################################
        watch_loss = []
        for batch in train_data:
            ####################################
            if cuda_flag:
                batch = [d.cuda() for d in batch]
            ####################################
            optim.zero_grad()  # 梯度归零
            index_seq, target_seq = batch
            mask = geneMask(config['batch_size'], config['txt1_len'], config['txt2_len'])
            loss = model.forward(index_seq, target_seq, mask)  # 计算loss
            watch_loss.append(loss.item())
            loss.backward()  # 反向传播
            optim.step()  # 梯度更新
        logger.info('第%d轮,平均loss:%f' % (epoch, np.mean(watch_loss)))
        logger.info(generate_sentence('阿根廷歹徒抢服装尺码不对拿回店里换', model, config['bert_path']))
        logger.info(generate_sentence('国际通用航空大会沈阳飞行家表演队一飞机发生坠机，伤亡不明', model, config['bert_path']))
    model_path = os.path.join(config["model_path"], "sft_model.pth")
    torch.save(model.state_dict(), model_path)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


def geneMask(batch_size, len1, len2):
    mask1 = torch.ones(batch_size, len1, len1)
    mask2 = torch.zeros(batch_size, len1, len2)
    mask3 = torch.ones(batch_size, len2, len1)
    mask4 = torch.tril(torch.ones(batch_size, len2, len2), diagonal=0)

    mask21 = torch.cat((mask1, mask2), dim=2)
    mask22 = torch.cat((mask3, mask4), dim=2)
    mask = torch.cat((mask21, mask22), dim=1)
    return mask


def generate_sentence(ask, model, bert_path):
    model.eval()
    with torch.no_grad():

        answer = ''
        pred_char = ''
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(answer) <= 100:
            x = tokenizer.encode(ask, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y_pred = model.forward(x)
            y = y_pred[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
            ask += pred_char
            answer += pred_char
    return answer


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


if __name__ == '__main__':
    from config import Config

    mian(Config)

    # print(geneMask(2, 3, 2))
