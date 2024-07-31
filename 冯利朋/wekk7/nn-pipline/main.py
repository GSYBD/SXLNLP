import torch.cuda
import pandas as pd
from model import TorchModel, choose_optimizer
from loader import load_data
from evaluate import Evaluate
import os
import logging
from config import Config
import numpy as np
import random
import csv
from transformers import BertTokenizer
from predict import Predict
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    epoch_num = config['epoch']
    train_data = load_data(config['train_data_path'], config)
    model = TorchModel(config)
    evaluate = Evaluate(config, model, logger)
    optimizer = choose_optimizer(config, model)
    if torch.cuda.is_available():
        model = model.cuda()
    for epoch in range(epoch_num):
        epoch += 1
        logger.info('第%d轮训练开始' % epoch)
        watch_loss = []
        model.train()
        for index, batch_data in enumerate(train_data):
            model.train()
            if torch.cuda.is_available():
                batch_data = [c.cuda() for c in batch_data]
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        logger.info("avg loss = %f" % (np.mean(watch_loss)))
        acc = evaluate.eval(epoch)
    model_path = os.path.join(config["model_path"], "%s---%d.pth" % (config['model_type'],random.randint(1,10000)))
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc, model, model_path

def write_csv(res,filename):
    config = res[0]
    fieldnames = list(config.keys())

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(res)
def config_to_result(config):
    result = {}
    result['Model'] = config['model_type']
    result['HiddenSize'] = config['hidden_size']
    result['LearningRate'] = config['learning_rate']
    result['BatchSize'] = config['batch_size']
    result['optimizer'] = config['optimizer']
    result['PoolStyle'] = config['pooling_style']
    result['输出模型名称'] = config['output_model_path']
    result['Acc'] = config['acc']
    result[f'time(预测耗时)'] = config['cost_time']
    return result
# 切分数据
def split_data(data_path, config):
    data = pd.read_csv(data_path)
    shuffled_indices = np.random.permutation(len(data))
    shuffled_data = data.iloc[shuffled_indices]
    # 划分数据集
    # 将70%的数据用于训练，15%用于验证，剩下的15%用于测试
    train_size = int(0.7 * len(shuffled_data))
    val_size = int(0.15 * len(shuffled_data))
    test_size = len(shuffled_data) - train_size - val_size

    # 划分数据集
    train_data = shuffled_data.iloc[:train_size]
    val_data = shuffled_data.iloc[train_size:train_size + val_size]
    test_data = shuffled_data.iloc[train_size + val_size:]

    train_data_path = config['train_data_path']
    valid_data_path = config['valid_data_path']
    test_data_path = config['test_data_path']

    train_data.to_csv(train_data_path, index=False)
    val_data.to_csv(valid_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    # print(train_data['label'].value_counts())
    # print(val_data['label'].value_counts())
    # print(test_data['label'].value_counts())

    # 长度统计
    # length_stats = train_data['review'].str.len().describe()
    # print(length_stats)
    # print("--------------")
    # length_stats = val_data['review'].str.len().describe()
    # print(length_stats)
    # print("--------------")
    # length_stats = test_data['review'].str.len().describe()
    # print(length_stats)
    # print("--------------")
# 评估文本长度

def ask(query, model_path,config,train_data):
    # input_ds = train_data.dataset.sentence_encode(query)
    # tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])
    # input_ds = tokenizer.encode(query, max_length=config['max_length'], pad_to_max_length=True)
    input_ds = train_data.dataset.sentence_encode(query)
    model = TorchModel(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        pred = model(torch.LongTensor([input_ds]))
    print(torch.argmax(pred))

if __name__ == '__main__':

    # 此方法用于分割数据集
    # split_data('./data/文本分类练习.csv', Config)

    # 单个模型预测
    # main(Config)

    # 预测
    train_data = load_data(Config['train_data_path'], Config)
    model_path = './output/lstm---8729.pth'
    ask('还不错,吃了好几次了，下次路过还过来', model_path, Config, train_data)

    # 矩阵网格，对几个模型进行评估
    # res = []
    # for model in ["fast_text", "gated_cnn", "lstm", "bert"]:
    #     if model == 'bert':
    #         Config["use_bert"] = True
    #     else:
    #         Config["use_bert"] = False
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-5]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg"]:
    #                     Config["pooling_style"] = pooling_style
    #                     # print("最后一轮准确率：", main(Config), "当前配置：", Config)
    #                     acc, model, model_path = main(Config)
    #                     predict = Predict(Config['test_data_path'], Config, model)
    #                     acc, cost_time = predict.predict()
    #                     Config['acc'] = acc
    #                     Config['output_model_path'] = model_path
    #                     Config['cost_time'] = cost_time
    #                     res.append(config_to_result(Config))
    # write_csv(res, 'result.csv')





