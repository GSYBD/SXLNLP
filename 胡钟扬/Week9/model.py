import torch
import re
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torchcrf import CRF

from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import json

from loader import load_vocab, encode_sentence
from typing import List


class ModelHub:
	'''
		choose your model to train
	'''
	def __init__(self, model_name, config):
		if model_name == "bert":
			self.model = BertCRFModel(config)
		elif model_name == "lstm":
			self.model = TorchModel(config)
		elif model_name == 'regex':
			self.model = RegularExpressionModel(config)
		elif model_name=='sentence':
			self.model = WholeSentenceNERModel(config)
		else:
			raise NotImplementedError("model name not supported")



class TorchModel(nn.Module):
	def __init__(self, config):	
		super(TorchModel,self).__init__()
		hidden_size = config["hidden_size"]
		# 必须先跑loader，获得vocab_size
		vocab_size = config["vocab_size"] + 1  # leave 0 for padding
		max_length = config["max_length"]
		class_num = config["class_num"]
		num_layers = config["num_layers"]
		self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
		self.bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
		self.classify = nn.Linear(hidden_size * 2, class_num) 
		# crf层, 用来计算 emission score tensor
		self.crf_layer = CRF(class_num, batch_first=True)
		self.use_crf = config["use_crf"]
		# -1 is the padding value for labels, which will be ignored in loss calculation
		self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

	def forward(self, x, target = None):
		'''
			loss: (batch_size * seq_len, 1)
  		'''
		x = self.embedding(x) # (batch_size, seq_len)
  
		x,_ = self.bilstm(x) # (batch_size, seq_len, hidden_size * 2)

		predict = self.classify(x) # (batch_size, seq_len, class_num)
		
		if target is not None:
			if self.use_crf:
				mask = target.gt(-1)
				# crf自带cross entropy loss
				# CRF loss 最后需要取反
				return - self.crf_layer(predict, target, mask, reduction = 'mean')
		 		
			else:
				return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
		else:
			if self.use_crf:
				# 维特比解码 viterbi
				return self.crf_layer.decode(predict) # (batch_size, seq_len)
			else:
				return predict

class BertCRFModel(nn.Module):
	'''
		基于BERT的CRF模型
	'''
	def __init__(self,config):
		super().__init__()
		self.config = {
			"bert_model_path":config["bert_config"]['bert_model_path'],
			"class_num":config['class_num'],
			"hidden_size":config['bert_config']['hidden_size'],
			"dropout":config['bert_config']['dropout']
		}
  
		self.bert = BertModel.from_pretrained(self.config["bert_model_path"], return_dict = False)
		self.classifier = nn.Linear(self.config["hidden_size"], self.config["class_num"])
		self.crf = CRF(self.config['class_num'], batch_first=True)
	def forward(self,x, target = None):
		sequence_output, _ = self.bert(x) # (batch_size, seq_len, hidden_size)
		# print("sequence_output = \n", sequence_output)
		
		predicts = self.classifier(sequence_output) # (batch_size, seq_len, class_num)

		if target!=None: # 计算CRF Loss
			mask = target.gt(-1)
			loss = self.crf(predicts, target, mask, reduction='mean') # (batch_size, seq_len, class_num)
			return -loss
		else:
			return self.crf.decode(predicts) # (batch_size, seq_len)
	
  
class WholeSentenceNERModel(nn.Module):
	'''
	  do the NER task for the entire sentence (sentence classification)
	'''
	def __init__(self, config):
		super().__init__()
		self.bert = BertModel.from_pretrained(config["bert_path"], return_dict = False)
		self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
		self.num_labels = config["sentence_config"]['num_labels']
		
		if config['sentence_config']['recurrent'] == "lstm":
			self.recurrent_layer = nn.LSTM(self.bert.config.hidden_size, 
                                  self.bert.config.hidden_size//2, 
                                  batch_first=True, 
                                  bidirectional=True,
                                  num_layers = 1)
		elif config['sentence_config']['recurrent'] == 'gru':
			self.recurrent_layer = nn.GRU(self.bert.config.hidden_size, 
                                 			self.bert.config.hidden_size//2,
											batch_first=True,
											bidirectional=True,
											num_layers =1
											)
		else:
			assert False
   
		
		self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
		
			
	
	def forward(self, input_ids=None, attention_mask=None, labels=None):
		'''
			input_ids: (batch_size, seq_len)
			attention_mask: (batch_size, seq_len)
			labels: (batch_size, seq_len)
  		'''
    
		output = self.bert(input_ids, attention_mask) # (batch_size, seq_len, hidden_size)

		pooled_output = output[1] # (batch_size, hidden_size)
  
		pooled_output = self.dropout(pooled_output)
  
		recurrent_output,_ = self.recurrent_layer(pooled_output.unsqueeze(0)) # (1, batch_size, hidden_size) 
		
		# 线性层只能处理二维张量
		output = self.classifier(recurrent_output.squeeze(0)) # (batch_size, num_labels)
  
  
		if labels is not None:
			loss = nn.CrossEntropyLoss()
			return loss(output, labels.view(-1))
		else:
			return output
  
class RegularExpressionModel(nn.Module):
	'''
		完全基于正则表达式的序列标注模型
		do sequence labeling with regular expression
	'''
	def __init__(self,config):
		super().__init__()
		self.config = config
		self.vocab = load_vocab(config["vocab_path"])
		self.reverse_vocab = self.load_reverse_vocab()
	
	def encode_char(self, char:str):
		return self.vocab.get(char, self.vocab['[UNK]'])
	def encode_sentence(self, text, padding = True):
		input_id = []
		
		for word in text:
			input_id.append(self.vocab.get(word,self.vocab['[UNK]']))
		return input_id

	def padding(self, input_id, pad_token=0):
		# padding
		input_id += [pad_token]*(self.config['max_length']-len(input_id))
		
		# truncate
		return input_id[:self.config['max_length']]
	
	def load_reverse_vocab(self):
		return {v:k for k,v in self.vocab.items()}

	def forward(self, x, target = None) -> List[List]:
		'''
			return [[entity1, entity2 ....], 
    				[entity3, entity4, ...]
		'''
		x # (batch_size, seq_len)
		pattern_list = []
		# load the NER dict
		schema = json.load(open(self.config['schema_path'], encoding='utf8'))
		
		# load the entity corpus
		entity_dict = []
		with open(Config['train_data_path'], encoding='utf8') as f:
			for line in f:
				line = line.strip()
				if line:
					line = line.split()
					entity_dict.append((line[0], line[1]))
		print("entity dict loaded, size = ", len(entity_dict))
	
		# print("entity dict = ", entity_dict)
		
		entity_pattern_dict = {}
		content = "" # 实体字符串
		entity = ""  # 实体类型
		for key, value in entity_dict:
			if value == 'O' and content=="":
				entity_pattern_dict[key] = value
				content = ""
				entity = value
			if value == 'O' and content!="":
				# 记录上一轮检测到的实体
				entity_pattern_dict[content] = entity
				content = ""
				entity = value
				# 记录本轮的无关字
				entity_pattern_dict[key] = value
			else:
				content += key
				entity = re.sub(r".*-", "",value)

		if content!="":
			entity_pattern_dict[content] = entity
		
  

		# 匹配
		entity_matrix = [] # 记录每行匹配到的实体
		for row in x:
			row = row.tolist()
			# 转为字符串
			row = "".join([self.reverse_vocab.get(i) for i in row])
			print("row = ", row)
			entity_row = [] # 记录匹配到的实体
			for pattern, value in entity_pattern_dict.items():
				# 取出pattern中的非字母数字下划线
				pattern2 = re.sub(r"[\W]", "", pattern)
				if pattern2 == "":
					continue
				# print("pattern = ", pattern)
				if re.search(pattern2,row) is not None:
					print("match: ", row, " ", pattern2)
					entity_row.append(entity_pattern_dict[pattern])
			entity_matrix.append(entity_row)
		return entity_matrix
     

class MyTokenizer(nn.Module):
    '''
     A self-defined tokenizer
    '''
    def __init__(self, config):
        pass

def choose_optimizer(config, model):
	optimizer = config['optimizer']
	learning_rate = config['learning_rate']
 
	if optimizer == 'adam':
		return Adam(model.parameters(), lr=learning_rate)
	elif optimizer == 'sgd':
		return SGD(model.parameters(), lr=learning_rate)
	elif optimizer == 'adamw':
		return AdamW(model.parameters(), lr=learning_rate)



def id_to_label(id, config):
	'''
	return label
	'''
	label2id = {}
	with open(config['schema_path'], 'r', encoding = 'utf8') as f:
		label2id = json.load(f)

	for k, v in label2id.items():
		if v == id:
			return k

		



if __name__ == '__main__':
	# from config import Config
	# model = TorchModel(Config)
	from config import Config
	
 
 
	model = RegularExpressionModel(Config)
	string = "筹集到海外侨胞捐资1800万元,全部用于发展平民医院的“硬件”建设。" 
	input = encode_sentence(string, Config)
	input = torch.LongTensor([input])
 
	print("input = \n",input)
 
	# output = model(input)
 
	# print(output)	
	# model = BertCRFModel(Config)
	# output = model(input)
	# print(output)	
	
 
	input_ids = torch.LongTensor([[1,3,34,67,64,678,123],[123,356,347,673,642,634,183]])
	attention_mask = torch.LongTensor([[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]])
	labels = torch.LongTensor([[1], [0]])
	model = WholeSentenceNERModel(Config)
	output = model(input_ids, attention_mask, labels)
	print(output)
  
