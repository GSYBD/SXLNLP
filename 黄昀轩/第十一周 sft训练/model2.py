import numpy as np
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch
import torch.functional
import json
import random
from torch.utils.data import DataLoader, Dataset
import os

class SFT(nn.Module):
	def __init__(self, vocab_size, hidden_size):
		super(SFT, self).__init__()
		self.bert = BertModel.from_pretrained('../bert-base-chinese', return_dict=False)
		self.classify = nn.Linear(hidden_size, vocab_size)
		self.loss = nn.CrossEntropyLoss(ignore_index=-1)
	
	def forward(self, x, mask=None, y=None):
		if y is not None:
			x, _ = self.bert(x, attention_mask=mask)
			y_pred = self.classify(x)
			return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1)) #注意这里要y_pred.shape[-1]
		else:
			x, _ = self.bert(x)
			y_pred = self.classify(x)
			return torch.softmax(y_pred, dim=-1)


def load_corpus():
	corpus = []
	with open('sample_data.json', encoding='utf8') as f:
		for line in f:
			item = json.loads(line)
			corpus.append([item['title'], item['content']])
	return corpus


def build_dataset(corpus, batch_size, tokenizer, window_size):
	data_set = []
	for prompt, answer in corpus:
		encode_prompt = tokenizer.encode(prompt, add_special_tokens=False)
		encode_answer = tokenizer.encode(answer, add_special_tokens=False)
		x = [tokenizer.cls_token_id] + encode_prompt + [tokenizer.sep_token_id] + encode_answer + [
			tokenizer.sep_token_id]
		y = [-1] * len(encode_prompt) + [-1] + encode_answer + [tokenizer.sep_token_id] + [-1]
		mask = build_mask(encode_prompt, encode_answer) #这里开始prompt错写成answer了 导致mask形状错误，导致出现 预测值一致重复的情况
		x = x[:window_size] + [0] * (window_size - len(x))
		y = y[:window_size] + [0] * (window_size - len(y))
		x = torch.LongTensor(x)
		y = torch.LongTensor(y)
		mask = pad_mask(mask, (window_size, window_size))
		data_set.append([x, mask, y])
	
	return DataLoader(data_set, batch_size, shuffle=True, num_workers=0)

#
def build_mask(s1, s2):
	len_s1 = len(s1) + 2
	len_s2 = len(s2) + 1
	mask = np.ones((len_s1 + len_s2, len_s1 + len_s2))
	for i in range(len_s1):
		mask[i, len_s1:] = 0 #mask[i, len_s1:]这里len_s1后面少写了：导致后面的掩码没有变成0 同样出现了mask失效 输出结果重复的情况
	for i in range(len_s2):
		mask[len_s1 + i, len_s1 + i + 1:] = 0
	return mask


def pad_mask(mask, size):
	hight, weight = mask.shape
	target_h, target_w = size
	pad_mask = np.zeros(size, dtype=mask.dtype)
	hight_start = 0
	weight_start = 0
	hight_end = min(target_h, hight)
	weight_end = min(target_w, weight)
	pad_mask[hight_start:hight_end, weight_start:weight_end] = mask[:hight_end, :weight_end]
	return pad_mask
# def build_mask(s1, s2):
#     len_s1 = len(s1) + 2 #cls + sep
#     len_s2 = len(s2) + 1 #sep
#     # 创建掩码张量
#     mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
#     # 遍历s1的每个token
#     for i in range(len_s1):
#         # s1的当前token不能看到s2的任何token
#         mask[i, len_s1:] = 0
#     # 遍历s2的每个token
#     for i in range(len_s2):
#         # s2的当前token不能看到后面的s2 token
#         mask[len_s1 + i, len_s1 + i + 1:] = 0
#     return mask
# def pad_mask(tensor, target_shape):
#     # 获取输入张量和目标形状的长宽
#     height, width = tensor.shape
#     target_height, target_width = target_shape
#     # 创建一个全零张量,形状为目标形状
#     result = torch.zeros(target_shape, dtype=tensor.dtype)
#     # 计算需要填充或截断的区域
#     h_start = 0
#     w_start = 0
#     h_end = min(height, target_height)
#     w_end = min(width, target_width)
#     # 将原始张量对应的部分填充到全零张量中
#     result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
#     return result

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
	
def predict(sentence,model,tokenizer):
	model.eval()
	openings = tokenizer.encode(sentence)
	with torch.no_grad():
		while len(openings) <= 50:
			x = torch.LongTensor([openings])
			if torch.cuda.is_available():
				x = x.cuda()
			y = model(x)[0][-1]
			index = sampling_strategy(y)
			openings.append(index)
	return tokenizer.decode(openings)
	
	

def main(corpus_path,save_weight=True):
	vocab_size = 21128
	hidden_size = 768
	epoch_nums = 20
	window_size = 50
	batch_size = 64
	learning_rate = 0.001
	save_weigth = False
	#设置模型、优化器
	tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')
	model = SFT(vocab_size, hidden_size)
	optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
	#创建训练集
	corpus = load_corpus()
	train_data = build_dataset(corpus, batch_size, tokenizer, window_size)
	# 训练模型
	model.train()
	watch_loss = []
	for epoch in range(epoch_nums):
		for x,mask,y in train_data:
			if torch.cuda.is_available():
				x,mask,y = x.cuda(),mask.cuda().y.cuda()
			optim.zero_grad()
			loss = model(x,mask,y)
			loss.backward()
			optim.step()
			watch_loss.append(loss.item())
		print('====第%d轮训练,平均loss:%f====' % (epoch + 1, np.mean(watch_loss)))
		print(predict("大学最优生物钟：每个年级该如何度过24小时？", model, tokenizer))
		print(predict("南京一合金厂锅炉发生爆炸", model, tokenizer))
	if not save_weight:
		return
	else:
		base_name = os.path.basename(corpus_path).replace("txt", "pth")
		model_path = os.path.join("model", base_name)
		torch.save(model.state_dict(), model_path)
		return
	
if __name__ == "__main__":
    main("sample_data.json",save_weight=False)
