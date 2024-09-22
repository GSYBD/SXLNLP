import torch
import torch.nn as nn
import os
import random
import numpy as np
import logging
from config import config
import evaluate
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]

logging.basicConfig(filename=f'Logger', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
date_format = '%Y-%M-%d %H:%M'
formatter = logging.Formatter(log_format, datefmt=date_format)
file_handeler = logging.FileHandler('out_put/Model_Logger.txt')
file_handeler.setFormatter(formatter)

"""
模型训练主程序
"""
seed = config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
	# 创建保存模型的目录
	if not os.path.isdir(config["model_path"]):
		os.mkdir(config["model_path"])
	# 加载训练数据
	train_data = load_data(config['train_data_path'], config)
	if train_data is not None:
		print('train_data_loaded,data_length:', len(train_data))
	model = TorchModel(config)
	model_ty = config['model_type']
	
	cuda_flag = torch.cuda.is_available()
	if cuda_flag:
		logger.info("GPU is available，change to cuda")
		model = model.cuda()
	optimizer = choose_optimizer(config, model)
	evaluator = Evaluator(config, model, logger)
	logger.info(f'===================Current_Model_Type:{model_ty}====================')
	for epoch in range(config['epoch']):
		epoch += 1
		model.train()
		train_loss = []
		logger.info("epoch %d begin" % epoch)
		for index, batch_data in enumerate(train_data):
			if cuda_flag:
				batch_data = [d.cuda() for d in batch_data]
			optimizer.zero_grad()
			inputs, labels = batch_data
			loss = model(inputs, labels)
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())
			if index % int(len(train_data) / 2 + 0.5) == 0:
				logger.info("batch loss %f" % loss)
		logger.info("epoch average loss: %f" % np.mean(train_loss))
		acc = evaluator.eval(epoch)
		logger.addHandler(file_handeler)
	
	return acc


if __name__ == '__main__':
	# main(config)
	for model_name in ['bert', 'lstm', 'gru', 'rnn cnn', ' gated_cnn', ' stack_gated_cnn ', 'rcnn', ' bert_cnn',
	                   'bert_mid_layer']:
		config["model_type"] = model_name
		
		print("当前配置：", config["model_type"], '最后一轮准确率：', main(config))
