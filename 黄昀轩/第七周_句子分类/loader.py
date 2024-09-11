import numpy as np
import tokenizers
import pandas as pd
import torch
import os
import re
import json
from torch.utils.data import DataLoader,Dataset
from transformers import BertTokenizer
from config import config

class DataGenerator:
	def __init__(self,config,data_path):
		self.path = data_path
		self.config = config
		self.index_to_label={
			0:"负面评价",
			1:"正面评价"
		}
		self.label_to_index = dict((y,x) for x,y in self.index_to_label.items())
		self.config['class_nums'] = len(self.index_to_label)
		if self.config['model_type'] == 'bert':
			self.tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_path'])
		self.vocab = load_vocab(config['vocab_path'])
		self.config['vocab_size'] = len(self.vocab)
		self.load()
	
	def load(self):
		self.data = []
		# with open(self.path,encoding='utf8') as f:
		# 	for line in f:
		# #line = json.loads(line)
		df = pd.read_csv(self.path) #用pandas导入数据
		print(f'从{self.path}中加载数据:')
		print('数据个数:',len(df))
		for i in range(len(df)):
			label = df.loc[i]['label']
			review = df.loc[i]['review']
			
			if self.config['model_type'] =='bert':
				input_id = self.tokenizer.encode(review,max_length= self.config['max_length'],pad_to_max_length=True) #这里没有pad_to_max
			else:
				input_id = self.encode_sentence(review)
			input_id = torch.LongTensor(input_id)
			label_index = torch.LongTensor([label])
			self.data.append([input_id,label_index]) #编码后的句子 及其标签
	
	def encode_sentence(self,text):
		input_id = []
		for char in text:
			input_id.append(self.vocab.get(char,self.vocab['[UNK]'])) #从字典中找单词对应数字
		input_id = self.padding(input_id)
		return input_id
	#截断/补齐至config['max_length']长度
	def padding(self,input_id):
		input_id = input_id[:self.config['max_length']]
		input_id +=[0]*(self.config['max_length']-len(input_id))
		return input_id
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		return self.data[index]
def load_vocab(vocab_path):
	token_dict= {}
	with open(vocab_path,encoding='utf8') as f:
		for index,line in enumerate(f):
			token = line.strip()
			token_dict[token] = index+1
	return token_dict
	
def load_data(data_path,config,shuffle=True):
	dg = DataGenerator(config,data_path)
	dl = DataLoader(dg,batch_size=config['batch_size'],shuffle=True)
	return dl
if __name__ == "__main__":

	dg1 = DataGenerator(config,'data/train.csv') #这里需要读取准备好的 数据（标注+句子）
	# print(dg[0:10])
	print(dg1[10][0].size())
	print(dg1[10][1].size())
	
	dg2 = DataGenerator(config,'data/test.csv') #这里需要读取准备好的 数据（标注+句子）
	# print(dg[0:10])
	print(dg2[0][0].size())
	print(dg2[0][1].size())
